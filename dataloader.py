import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class CustomDataset(Dataset):
    def __init__(self, folder_path, split):
        self.folder_path = folder_path
        self.split = split
        self.data = self.load_data()
        self.label_encoding = {'I': 0, 'O': 1, 'P': 2, 'S': 3, 'M':4, 'B': 5}

    def load_data(self):
        file_list = [f for f in os.listdir(self.folder_path) if f.endswith('.npy')]
        file_list.sort()  # Make sure the order is consistent

        data = []
        for file_name in file_list:
            file_path = os.path.join(self.folder_path, file_name)
            data.append(np.load(file_path, allow_pickle=True).item())

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        #print(sample['data'])
        inputs = sample['data']
        labels_str = sample['labels']

        labels_list = [self.label_encoding[label] for label in labels_str]
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        return {'data': inputs, 'labels': labels_tensor}

def collate_fn(batch):
    # Sort the batch by sequence length in descending order
    batch = sorted(batch, key=lambda x: len(x['data']), reverse=True)

    # Pad sequences for data
    data = [torch.tensor(sample['data']) for sample in batch]
    padded_data = pad_sequence(data, batch_first=True)

    # Pad sequences for labels
    labels = [torch.tensor(sample['labels']) for sample in batch]
    padded_labels = pad_sequence(labels, batch_first=True)

    # Pack the padded sequences for data
    lengths = [len(seq) for seq in data]
    packed_data = pack_padded_sequence(padded_data, lengths=lengths, batch_first=True, enforce_sorted=False)

    return {'data': packed_data, 'labels': padded_labels} 

def create_data_loaders(data_root):
    splits = [
        (['cv0', 'cv1', 'cv2'], 'cv3' , 'cv4'),
        (['cv1', 'cv2', 'cv3',], 'cv4', 'cv0'),
        (['cv2', 'cv3', 'cv4'], 'cv0', 'cv1'),
        (['cv3', 'cv4', 'cv0'], 'cv1', 'cv2'),
        (['cv4', 'cv0', 'cv1'], 'cv2', 'cv3'),
    ]

    data_loaders = {}

    for train_folders, val_folder, test_folder in splits:
        train_dataset = [CustomDataset(os.path.join(data_root, folder), 'train') for folder in train_folders]
        val_dataset = CustomDataset(os.path.join(data_root, val_folder), 'val')
        test_dataset = CustomDataset(os.path.join(data_root, test_folder), 'test')
        
        for train_folders, train_dataset in zip(train_folders, train_dataset):
            data_loaders[train_folders] = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        data_loaders[val_folder] = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        data_loaders[test_folder] = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    return data_loaders

if __name__ == "__main__":
    data_root = os.getcwd() + '/encoder_proteins_test'
    data_loaders = create_data_loaders(data_root)

    # Accessing the data loaders
    train_loader_cv0 = data_loaders['cv0']
    val_loader_cv1 = data_loaders['cv1']
    test_loader_cv2 = data_loaders['cv2']

    # Iterate through a few batches to test the DataLoader
    for batch_idx, batch in enumerate(train_loader_cv0):
        inputs, labels = batch['data'], batch['labels']

        print(f"Batch {batch_idx + 1}:")
        #print("Inputs shape:", inputs.shape)  # Assuming inputs is a NumPy array or a PyTorch tensor
        print("Labels shape:", labels.shape)  # Assuming labels is a NumPy array or a PyTorch tensor

        print(labels)
        # Break the loop after a few batches for testing purposes
        if batch_idx == 3:
            break