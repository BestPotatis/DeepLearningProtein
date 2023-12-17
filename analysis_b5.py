#!/usr/bin/env python
# coding: utf-8

from itertools import cycle, islice
import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import json
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from pytorchtools import EarlyStopping
from classifier_rnn import RNN
from accuracy_b5 import test_predictions
from accuracy_b5 import type_from_labels

class CustomDataset(Dataset):
    def __init__(self, folder_path, split):
        self.folder_path = folder_path
        self.split = split
        self.data = self.load_data()
        
        self.label_encoding = {'I': 0, 'O': 1, 'P': 2, 'S': 3, 'M': 4, 'B': 5}

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
        inputs = torch.tensor(sample['data'])
        labels_str = sample['labels']

        labels_list = [self.label_encoding[label] for label in labels_str]
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        
        p_type = type_from_labels(labels_tensor)

        return {'data': inputs, 'labels': labels_tensor, 'type': p_type}

def collate_fn(batch):
    # sort the batch by sequence length in descending order
    batch = sorted(batch, key=lambda x: len(x['data']), reverse=True)
    
    # pad sequences for data
    data = [torch.tensor(sample['data']) for sample in batch]
    padded_data = pad_sequence(data, batch_first=True, padding_value=-1)

    # pad sequences for labels
    labels = [torch.tensor(sample['labels']) for sample in batch]
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    
    # pack the padded sequences for data
    lengths = torch.tensor([len(seq) for seq in data])
    #packed_data = pack_padded_sequence(padded_data, lengths=lengths, batch_first=True, enforce_sorted=True)

    types = torch.tensor([sample["type"] for sample in batch])
    
    return {'data': padded_data, 'labels': padded_labels, "lengths": lengths, "type": types} 


def train_nn(model, trainloader, valloader, loss_function, optimizer, fold, experiment_file_path, device, num_epochs = 50, patience = 15):

    model.train()
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, path = f"best_model_{model.hidden_size}.pt")
    train_loss = []
    valid_loss = []
    top_acc = []
    
    for epoch in range(num_epochs):
        for _, batch in enumerate(trainloader):
            inputs, labels, lengths, types = batch['data'], batch['labels'], batch['lengths'], batch["type"]
            inputs, labels, lengths, types = inputs.to(device), labels.to(device), lengths.to(device), types.to(device)
            
            # receive output from rnn
            output = model(inputs)  
            
            # loss = loss_function(output.permute(0, 2, 1), labels)
            
            loss = 0    
            for l in range(output.shape[0]):
                # masking the zero-padded outputs
                batch_output = output[l][:lengths[l]]
                batch_labels = labels[l][:lengths[l]]
                
                # compute cross-entropy loss
                loss += loss_function(batch_output, batch_labels) / output.shape[0]
            
            # gradient update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
        
        train_loss_epoch, _ = test_predictions(model = model, loader = trainloader, loss_function = loss_function,
                         cv = str(fold), experiment_file_path = experiment_file_path, condition = "train", epoch = str(epoch),
                         device = device)   

        valid_loss_epoch, valid_top_acc = test_predictions(model = model, loader = valloader, loss_function = loss_function,
                        cv = str(fold), experiment_file_path = experiment_file_path, condition = "val", epoch = str(epoch),
                        device = device)
        
        # receive mean loss for this epoch
        train_loss.append(train_loss_epoch)
        valid_loss.append(valid_loss_epoch)
        top_acc.append(valid_top_acc)
        
        # checking val loss for early stopping
        early_stopping(valid_top_acc, model)
        
        if early_stopping.early_stop:
            # print("Early stopping, best loss: ", train_loss[-16], -early_stopping.best_score)
            
            return train_loss[-16], -early_stopping.best_score
        
        # if epoch % 1 == 0:
        #     print("training loss: ", train_loss[-1], \
        #         "\t validation loss: ", valid_loss[-1],
        #         "\t early stop score:", -early_stopping.best_score)
    
    return train_loss[-1], top_acc[-1]


if __name__ == "__main__":
    # set device to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")

    # config
    k_folds = 5
    num_epochs = 100
    batch_size = 32
    loss_function = nn.CrossEntropyLoss(ignore_index = -1)
    lr = 1e-3
    tuning = [64, 128, 256, 512]
    encoder_path = "encoder_proteins"

    experiment_file_list = []
    for i in tuning:
        experiment_file_list.append(f"stat_data_B5_{i}.json")
        experiment_json = {}
        open(experiment_file_list[-1], 'w').write(json.dumps(experiment_json))

    # set fixed random number seed
    torch.manual_seed(42)

    # create cv's corresponding to deeptmhmm's
    cvs = list(range(5))
    kfolds = []
    for idx, split in enumerate(range(5)):
        
        # make cycling list and define train/val/test splits
        idxs = np.asarray(list(islice(cycle(cvs), idx, idx + 5)))
        train_idx, val_idx, test_idx = idxs[:3], idxs[3], idxs[4]
        
        kfolds.append((train_idx, val_idx, test_idx))

    # make on big concatenated dataset of all splits
    data_cvs = np.squeeze([CustomDataset(os.path.join(encoder_path, folder), 'train') for folder in ['cv0', 'cv1', 'cv2', 'cv3' , 'cv4']])

    # k-fold cross validation
    for fold, (train_ids, val_id, test_id) in enumerate(kfolds):    
        # print(f'\nFOLD {fold + 1}')
        # print('--------------------------------')
        
        # concatenates the data from the different cv's
        training_data = np.concatenate(data_cvs[train_ids], axis = 0)
        
        # create weighted sampler with replacement to class balance
        y_train = torch.tensor([prot_type["type"] for prot_type in training_data])
        
        weight = torch.zeros(5)
        for t in np.unique(y_train):
            class_sample_count = torch.tensor([len(torch.where(y_train == t)[0])])
            weight[t] = 1. / class_sample_count
        
        samples_weight = torch.tensor([weight[t] for t in y_train])
        
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement = True, generator = torch.Generator(device = device))
        
        # define data loaders for train/val/test data in this fold (collate 0 pads for same-length)
        trainloader = DataLoader(
                            training_data, batch_size=batch_size, collate_fn=collate_fn, drop_last = False, generator = torch.Generator(device = device), sampler = sampler)

        valloader = DataLoader(data_cvs[val_id], batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last = False)
        testloader = DataLoader(data_cvs[test_id], batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last = False)
        
        val_acc_param = np.zeros((len(tuning)))
        
        # hyperparameter tune
        for idx, param in enumerate(tuning):
            experiment_file_path = experiment_file_list[idx] 
            
            # print(f'\nHIDDEN_SIZE {param}')
            # print('--------------------------------')
            
            # define models to be analyzed
            model_rnn = RNN(512, param, 6)
            model_rnn.to(device)
            
            optimizer = optim.Adam(model_rnn.parameters(), lr = lr)
            
            # train and validate model
            train_loss, valid_top_acc = train_nn(model = model_rnn, 
                                            trainloader = trainloader, valloader = valloader,
                                            loss_function = loss_function, optimizer = optimizer, 
                                            fold = fold, experiment_file_path = experiment_file_path, 
                                            device = device, num_epochs = num_epochs)
            
            # save topology accuracies
            val_acc_param[idx] = valid_top_acc
        
        # test for the best model
        best_param_idx = val_acc_param.argmax()
        
        best_model = RNN(512, tuning[best_param_idx], 6)
        best_model.load_state_dict(torch.load(f"best_model_{tuning[best_param_idx]}.pt"))
        best_model.eval()
        
        experiment_file_path = experiment_file_list[best_param_idx]
        
        # print(f"\nbest params for fold {fold + 1}: ", tuning[best_param_idx])  
        
        test_loss, test_top_acc = test_predictions(model = best_model,
                        loader = testloader,
                        loss_function = loss_function,
                        cv = str(fold), experiment_file_path = experiment_file_path,
                        device = device)

        # print(f"test loss for fold {fold + 1}: ", test_loss, \
        #     f"\ntest topology accuracy for fold {fold + 1}: ", test_top_acc)