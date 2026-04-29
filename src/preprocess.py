import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


def load_train_data_with_val(val_ratio=0.1):
    train_data = pd.read_csv('data/train.csv')
    
    labels = train_data['label'].values
    features = train_data.drop('label', axis=1).values
    
    features = features / 255.0
    
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(features, labels)
    
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader


def load_test_data():
    test_data = pd.read_csv('data/test.csv')
    
    features = test_data.values
    features = features / 255.0
    
    features = torch.tensor(features, dtype=torch.float32)
    
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    return loader
