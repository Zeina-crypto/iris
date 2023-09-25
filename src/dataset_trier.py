import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.tensor(np.load(file_path), dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        return sample

# Paths to your numpy array files
train_file_path = '/space/ankushroy/Data/all_data_train.npy'
test_file_path = '/space/ankushroy/Data/all_data_test.npy'
vali_file_path = '/space/ankushroy/Data/all_data_vali.npy'

# Create instances of the dataset for training, testing, and validation
train_dataset = CustomDataset(train_file_path)
test_dataset = CustomDataset(test_file_path)
vali_dataset = CustomDataset(vali_file_path)

# Create DataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
vali_loader = DataLoader(vali_dataset, batch_size=8, shuffle=False)

# Get the first batch of data
first_batch = next(iter(train_loader))

# Print the shape of the first batch
print("Shape of the first batch:", first_batch.shape)