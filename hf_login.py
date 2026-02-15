from huggingface_hub import login
import getpass

token = getpass.getpass("Enter your HuggingFace token: ")
login(token=token)
print("Login successful!")
