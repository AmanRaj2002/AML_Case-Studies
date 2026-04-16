import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os

def generate_raw_data(filepath='data/raw_data.csv', num_samples=25000, num_features=40):
    os.makedirs('data', exist_ok=True)
    np.random.seed(42)
    features = np.random.randn(num_samples, num_features)
    baseline_eta = np.random.uniform(300, 3600, size=num_samples)
    hidden_influence = np.sum(features, axis=1) * 3.0
    true_residual = hidden_influence + np.random.randn(num_samples) * 15
    actual_time = baseline_eta + true_residual
    
    columns = [f'feature_{i}' for i in range(num_features)] + ['baseline_eta', 'actual_time']
    df_raw = pd.DataFrame(np.column_stack((features, baseline_eta, actual_time)), columns=columns)
    df_raw.to_csv(filepath, index=False)
    return df_raw

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df['true_residual'] = df['actual_time'] - df['baseline_eta']
    return df

def split_data(df, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    train_df, temp_df = train_test_split(df, test_size=(val_frac + test_frac), random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=(test_frac / (val_frac + test_frac)), random_state=42)
    
    # Drop residual helper to prevent target leakage
    return train_df.drop(columns=['true_residual']), val_df.drop(columns=['true_residual']), test_df.drop(columns=['true_residual'])

def store_splits(train_df, val_df, test_df, output_dir='data'):
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/validation.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)

def create_dataloader(filepath, batch_size=512, shuffle=False):
    df = pd.read_csv(filepath)
    X = torch.tensor(df.iloc[:, :-2].values, dtype=torch.float32).view(-1, 5, 8) 
    baseline_eta = torch.tensor(df['baseline_eta'].values, dtype=torch.float32).unsqueeze(1)
    actual_time = torch.tensor(df['actual_time'].values, dtype=torch.float32).unsqueeze(1)
    return DataLoader(TensorDataset(X, baseline_eta, actual_time), batch_size=batch_size, shuffle=shuffle)