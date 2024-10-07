import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.preprocessing import StandardScaler
import os

def main(input_filepath='./data/processed', output_filepath='./data/processed'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    fX_test = f"{input_filepath}/X_test.csv"
    fX_train = f"{input_filepath}/X_train.csv"
    
    process_data(fX_test, fX_train, output_filepath)

def process_data(fX_test, fX_train,output_filepath):
    # Import datasets
    X_test = import_dataset(fX_test).drop(columns=["date"])
    X_train = import_dataset(fX_train).drop(columns=["date"])
    scaler = StandardScaler()
    print(X_train.head())
    
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    # Create folder if necessary
    create_folder_if_necessary(output_filepath)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train_scaled, X_test_scaled, output_filepath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if os.path.exists(output_folderpath)==False:
        os.makedirs(output_folderpath)

def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        #file.to_csv(output_filepath, index=False)
        np.savetxt(output_filepath, file, delimiter=",")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()