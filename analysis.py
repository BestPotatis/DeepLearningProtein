#!/usr/bin/env python
# coding: utf-8

from itertools import cycle, islice
import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from collections import defaultdict
from pytorchtools import EarlyStopping
from classifier_rnn import RNN
from accuracy import test_predictions


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

        return {'data': inputs, 'labels': labels_tensor}

def collate_fn(batch):
    # sort the batches by length
    batch = sorted(batch, key=lambda x: len(x['data']), reverse=True)
    
    # pad sequences
    data = [torch.tensor(sample['data']) for sample in batch]
    padded_data = pad_sequence(data, batch_first=True)

    # pad sequences for labels
    labels = [torch.tensor(sample['labels']) for sample in batch]
    padded_labels = pad_sequence(labels, batch_first=True)
    
    # pack the padded sequences for data
    lengths = [len(seq) for seq in data]
    #packed_data = pack_padded_sequence(padded_data, lengths=lengths, batch_first=True, enforce_sorted=True)

    return {'data': padded_data, 'labels': padded_labels, "lengths": lengths} 


def train_nn(model, trainloader, valloader, loss_function, optimizer, fold, experiment_file_path, device, num_epochs = 50, patience = 15):

    model.train()
    
    early_stopping = EarlyStopping(patience=patience, verbose=False)
    train_loss = []
    valid_loss = []
    
    for epoch in range(num_epochs):
        for _, batch in enumerate(trainloader):
            inputs, labels, lengths = batch['data'], batch['labels'], torch.tensor(batch['lengths'])
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            
            # receive output from rnn
            output = model(inputs)  
            
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
        
        train_loss_epoch = test_predictions(model = model, loader = trainloader, loss_function = loss_function,
                         cv = str(fold), experiment_file_path = experiment_file_path, condition = "train", epoch = str(epoch),
                         device = device)   

        valid_loss_epoch = test_predictions(model = model, loader = valloader, loss_function = loss_function,
                        cv = str(fold), experiment_file_path = experiment_file_path, condition = "val", epoch = str(epoch),
                        device = device)
        
        # # receive mean loss for this epoch
        train_loss.append(train_loss_epoch)
        valid_loss.append(valid_loss_epoch)
        
        # checking val loss for early stopping
        early_stopping(valid_loss_epoch, model)
        
        if early_stopping.early_stop:
            # print("Early stopping, best loss: ", train_loss[-16], -early_stopping.best_score)
            
            return train_loss[-16], -early_stopping.best_score
        
        # if epoch % 1 == 0:
        #     print("training loss: ", train_loss[-1], \
        #         "\t validation loss: ", valid_loss[-1],
        #         "\t early stop score:", -early_stopping.best_score)
    
    return train_loss[-1], valid_loss[-1]


if __name__ == "__main__":
    # set device to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")

    # config
    k_folds = 5
    num_epochs = 100
    batch_size = 32
    loss_function = nn.CrossEntropyLoss()
    lr = 1e-3
    tuning = [64, 128, 256, 512]
    encoder_path = "encoder_proteins"

    experiment_file_list = []
    for i in tuning:
        experiment_file_list.append(f"stat_data_{i}.json")
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

    gen_train_loss = np.zeros((len(kfolds), num_epochs))
    gen_train_acc = np.zeros((len(kfolds), num_epochs))
    fold_test_loss = np.zeros((len(kfolds)))
    fold_test_acc = np.zeros((len(kfolds)))

    # k-fold cross validation
    for fold, (train_ids, val_id, test_id) in enumerate(kfolds):    
        # print(f'\nFOLD {fold + 1}')
        # print('--------------------------------')
        
        # concatenates the data from the different cv's
        training_data = np.concatenate(data_cvs[train_ids], axis = 0)
        
        # define data loaders for train/val/test data in this fold (collate 0 pads for same-length)
        trainloader = DataLoader(
                            training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last = False, generator = torch.Generator(device = device))

        valloader = DataLoader(data_cvs[val_id], batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last = False)
        testloader = DataLoader(data_cvs[test_id], batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last = False)
        
        param_models = []
        
        train_loss_param = np.zeros((len(tuning)))
        val_loss_param = np.zeros((len(tuning)))
        
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
            train_loss, valid_loss = train_nn(model = model_rnn, 
                                            trainloader = trainloader, valloader = valloader,
                                            loss_function = loss_function, optimizer = optimizer, 
                                            fold = fold, experiment_file_path = experiment_file_path, 
                                            device = device, num_epochs = num_epochs)
            
            # save models and losses
            param_models.append(model_rnn)   
            train_loss_param[idx] = train_loss
            val_loss_param[idx] = valid_loss
        
        # test for the best model
        best_param_idx = val_loss_param.argmin()
        best_model = param_models[best_param_idx]
        experiment_file_path = experiment_file_list[best_param_idx]
        
        # print(f"\nbest params for fold {fold + 1}: ", tuning[best_param_idx])  
        
        test_loss = test_predictions(model = best_model,
                        loader = testloader,
                        loss_function = loss_function,
                        cv = str(fold), experiment_file_path = experiment_file_path,
                        device = device)

        # print(f"test loss for fold {fold + 1}: ", test_loss)