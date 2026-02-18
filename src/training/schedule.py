def freeze_text_encoder(model):

    for name, param in model.named_parameters():

        if "text_encoder" in name:
            param.requires_grad = False


def unfreeze_text_encoder_top(model):

    for name, param in model.named_parameters():

        if "text_encoder.layers.8" in name \
        or "text_encoder.layers.9" in name:
            param.requires_grad = True


def unfreeze_all(model):

    for param in model.parameters():
        param.requires_grad = True


def apply_stage(stage, model):

    if stage == 1:

        freeze_text_encoder(model)

    elif stage == 2:

        freeze_text_encoder(model)
        unfreeze_text_encoder_top(model)

    elif stage == 3:

        unfreeze_all(model)

    else:

        raise ValueError("Invalid stage")
