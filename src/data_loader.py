from pathlib import Path
import pandas as pd
import urllib.request

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset destination
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# MELD dataset CSV files
FILES = {
    "train_sent_emo.csv": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev_sent_emo.csv": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test_sent_emo.csv": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv"
}

def download_dataset():

    for filename, url in FILES.items():

        file_path = DATA_DIR / filename

        if file_path.exists():
            print(f"Already downloaded: {filename}")
            continue

        print(f"Downloading {filename}...")

        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded successfully: {filename}")

        except Exception as e:
            print(f"Error downloading {filename}: {e}")

    print("\nDataset download process completed!")


def load_dataset():

    train_df = pd.read_csv(DATA_DIR / "train_sent_emo.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev_sent_emo.csv")
    test_df = pd.read_csv(DATA_DIR / "test_sent_emo.csv")

    return train_df, dev_df, test_df


if __name__ == "__main__":

    download_dataset()

    train_df, dev_df, test_df = load_dataset()

    print("\n========== DATASET INFORMATION ==========")

    print("Training data:", train_df.shape)
    print("Validation data:", dev_df.shape)
    print("Testing data:", test_df.shape)

    print("\nTraining dataset columns:")
    print(train_df.columns.tolist())

    print("\nFirst 5 training records:")
    print(train_df.head())