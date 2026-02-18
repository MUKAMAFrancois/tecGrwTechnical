import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.cuda.amp import GradScaler, autocast

from src.training.schedule import apply_stage
from src.training.evaluation import run_eval
from src.training.checkpoint import save_checkpoint


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(
                param.detach(),
                alpha=1.0 - self.decay
            )

    def store(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()

    def copy_to(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}


class Trainer:

    def __init__(
        self,
        model,
        tokenizer,
        train_loader,
        val_loader,
        config,
        device,
        use_amp=True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.global_step = 0
        self.use_amp = use_amp and device.type == "cuda"

        self.scaler = GradScaler() if self.use_amp else None

        self.optimizer = None
        self.scheduler = None

        self.model.to(device)

        self.ema_enabled = bool(config.get("EMA_ENABLED", False))
        self.ema_decay = float(config.get("EMA_DECAY", 0.999))
        self.ema = ModelEMA(model, self.ema_decay) if self.ema_enabled else None

    # ----------------------------------
    # Stage setup
    # ----------------------------------

    def setup_stage(self, stage, lr):
        """
        Apply freezing policy and create optimizer.
        """

        apply_stage(stage, self.model)

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

        self.model.train()

    def setup_scheduler(self, total_steps):
        """
        Build a warmup + cosine schedule at batch-step granularity.
        """

        warmup_steps = int(self.config.get("WARMUP_STEPS", 0))
        warmup_steps = max(0, min(warmup_steps, total_steps))

        if total_steps <= 0:
            self.scheduler = None
            return

        if warmup_steps == 0:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=float(self.config.get("COSINE_MIN_LR", 0.0))
            )
            return

        if warmup_steps >= total_steps:
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=total_steps
            )
            return

        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=float(self.config.get("COSINE_MIN_LR", 0.0))
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps]
        )

    # ----------------------------------
    # Single training step
    # ----------------------------------

    def train_one_batch(self, batch):

        self.global_step += 1

        input_ids = batch["input_ids"].to(self.device)
        waveforms = batch["waveforms"].to(self.device)

        self.optimizer.zero_grad()

        if self.use_amp:

            with autocast():

                outputs = self.model(
                    input_ids=input_ids,
                    labels=waveforms
                )

                loss = outputs.loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:

            outputs = self.model(
                input_ids=input_ids,
                labels=waveforms
            )

            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0
            )

            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        if self.ema is not None:
            self.ema.update(self.model)

        log_every = int(self.config.get("LOG_EVERY", 50))
        if log_every > 0 and self.global_step % log_every == 0:
            print(f"step={self.global_step} loss={loss.item():.4f}")

        return loss.item()

    # ----------------------------------
    # Full epoch
    # ----------------------------------

    def train_one_epoch(self):

        total_loss = 0

        for batch in self.train_loader:

            loss = self.train_one_batch(batch)

            total_loss += loss

        if len(self.train_loader) == 0:
            return 0.0

        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    def _with_ema_weights(self):
        return self.ema is not None and bool(self.config.get("EVAL_USE_EMA", True))

    def _swap_in_ema(self):
        if not self._with_ema_weights():
            return
        self.ema.store(self.model)
        self.ema.copy_to(self.model)

    def _restore_from_ema(self):
        if not self._with_ema_weights():
            return
        self.ema.restore(self.model)

    @torch.no_grad()
    def validate(self):

        if self.val_loader is None:
            return None

        self.model.eval()
        self._swap_in_ema()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            waveforms = batch["waveforms"].to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=waveforms
                    )
                    loss = outputs.loss
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    labels=waveforms
                )
                loss = outputs.loss

            total_loss += float(loss.item())
            num_batches += 1

        self._restore_from_ema()
        self.model.train()

        if num_batches == 0:
            return None

        return total_loss / num_batches

    # ----------------------------------
    # Evaluation
    # ----------------------------------

    def evaluate(self, output_dir):

        self.model.eval()
        self._swap_in_ema()

        run_eval(
            self.model,
            self.tokenizer,
            self.device,
            output_dir,
            self.global_step
        )

        self._restore_from_ema()
        self.model.train()

    def save_model_checkpoint(self, output_dir):
        self.model.eval()
        use_ema_for_ckpt = bool(self.config.get("CHECKPOINT_USE_EMA", True))
        if use_ema_for_ckpt and self.ema is not None:
            self.ema.store(self.model)
            self.ema.copy_to(self.model)

        save_checkpoint(
            self.model,
            self.tokenizer,
            output_dir,
            self.global_step
        )

        if use_ema_for_ckpt and self.ema is not None:
            self.ema.restore(self.model)
        self.model.train()


    def train_stage(self, stage, lr, epochs):
        """
        Train one stage with freezing policy and cosine LR schedule.
        """

        self.setup_stage(stage, lr)

        total_steps = epochs * len(self.train_loader)
        self.setup_scheduler(total_steps)

        eval_dir = self.config.get("EVAL_DIR", "eval")
        ckpt_dir = self.config.get("CHECKPOINT_DIR", "checkpoints")

        print(f"\n[Stage {stage}] lr={lr} epochs={epochs}")

        for epoch in range(1, epochs + 1):

            avg_loss = self.train_one_epoch()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"[Stage {stage}] "
                f"epoch {epoch}/{epochs} "
                f"loss={avg_loss:.4f} "
                f"lr={current_lr:.8f}"
            )

            val_loss = self.validate()
            if val_loss is not None:
                print(
                    f"[Stage {stage}] "
                    f"epoch {epoch}/{epochs} "
                    f"val_loss={val_loss:.4f}"
                )

            self.evaluate(eval_dir)
            self.save_model_checkpoint(ckpt_dir)
