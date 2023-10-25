import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader





class Collector: 

    def __init__(self):
        self.training_data = []
        self.testing_data = []
        self.eval_data = []    

    class CustomDataset(Dataset):
        def __init__(self, file_path):
            self.data = torch.tensor(np.load(file_path), dtype=torch.float32)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            sample = self.data[index]
            return sample
        
    def collect_training_data(self, batch_size):
        train_file_path = '/space/zboucher/Data/all_data_train.npy'
        train_dataset = self.CustomDataset(train_file_path)
        self.training_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        length= len(train_dataset)

        return self.training_data, length
    
    def collect_testing_data(self, batch_size):
        test_file_path = '/space/zboucher/Data/all_data_test.npy'
        test_dataset = self.CustomDataset(test_file_path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        length=len(test_dataset)

        return test_loader, length
    
    def collect_validation_data(self):
        vali_file_path = '/space/zboucher/Data/all_data_vali.npy'

        vali_dataset = self.CustomDataset(vali_file_path)
        vali_loader = DataLoader(vali_dataset, batch_size=1, shuffle=False)
        length= len(vali_dataset)
        return vali_loader, length
    
    
    