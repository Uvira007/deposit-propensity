"""
Load Bank marketing dataset. Download from UCI if not present locally.
"""
from pathlib import Path
from zipfile import ZipFile
import urllib.request

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
ZIP_CSV_NAME = "bank-full.csv"


def _download_uci_bank(data_dir: Path, filename: str) -> Path:
    """
    Download and extract the bank marketing dataset from UCI.
    Return Path to the downloaded csv
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "bank.zip"
    # Download the zip file
    if not zip_path.exists():
        urllib.request.urlretrieve(UCI_URL, zip_path)
    # Extract the specified CSV file
    with ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(ZIP_CSV_NAME)as src:
             output_path = data_dir / filename
             with open(output_path, 'wb') as dst:
                 dst.write(src.read())
    return data_dir / filename

def load_bank_marketing(
        data_dir: Path = Path("data/raw"),
        filename: str = "bank-full.csv",
        download_if_missing: bool = True) -> pd.DataFrame:
        """
        Load bank marketing data.
        if dataset not present, download from UCI
        """
        data_dir = Path(data_dir)
        csv_path = data_dir / filename
        if not csv_path.exists():
            if not download_if_missing:
                raise FileNotFoundError(f"{csv_path} not found and download_if_missing is False.")
            csv_path = _download_uci_bank(data_dir, filename)
        # CSV uses semicolon separator
        return pd.read_csv(csv_path, sep=";")

if __name__ == "__main__":
     dataset = load_bank_marketing()
     print(dataset.head())