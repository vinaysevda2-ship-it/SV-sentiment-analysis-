import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# ---------------------------
# SAFE LOGGING SETUP (FIXED)
# ---------------------------
logger = logging.getLogger('data_ingestion')

# Prevent duplicate handlers (IMPORTANT FIX)
if not logger.handlers:
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('errors.log')
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ---------------------------
# LOAD PARAMS
# ---------------------------
def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logger.debug('Parameters retrieved from %s', params_path)
        return params

    except Exception as e:
        logger.error('Error loading params: %s', e)
        raise


# ---------------------------
# LOAD DATA
# ---------------------------
def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df

    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise


# ---------------------------
# PREPROCESS
# ---------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        df = df[df['clean_comment'].str.strip() != '']

        logger.debug('Data preprocessing completed.')
        return df

    except Exception as e:
        logger.error('Preprocessing error: %s', e)
        raise


# ---------------------------
# SAVE DATA
# ---------------------------
def save_data(train_data, test_data, data_path):
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug('Data saved at %s', raw_data_path)

    except Exception as e:
        logger.error('Save error: %s', e)
        raise


# ---------------------------
# MAIN PIPELINE (SAFE ENTRY POINT)
# ---------------------------
def main():
    logger.debug("===== DATA INGESTION STARTED =====")

    params_path = os.path.join(os.path.dirname(__file__), '../../params.yaml')
    params = load_params(params_path)

    test_size = params['data_ingestion']['test_size']

    df = load_data(
        'https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv'
    )

    final_df = preprocess_data(df)

    train_data, test_data = train_test_split(
        final_df,
        test_size=test_size,
        random_state=42
    )

    save_data(
        train_data,
        test_data,
        os.path.join(os.path.dirname(__file__), '../../data')
    )

    logger.debug("===== DATA INGESTION COMPLETED =====")


# ---------------------------
# EXECUTION GUARD (CRITICAL FIX)
# ---------------------------
if __name__ == "__main__":
    main()