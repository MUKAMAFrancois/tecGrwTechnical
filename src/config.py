import os

HF_DATASET_ID = "Washere-1/tecgrw-audio"

TARGET_SAMPLE_RATE = 16000  
MAX_DURATION = 12.0        
MIN_DURATION = 2.0          
STRIP_SILENCE = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)