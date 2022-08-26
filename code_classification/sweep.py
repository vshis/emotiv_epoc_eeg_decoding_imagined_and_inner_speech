import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
import wandb
from cnn import EEGNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {torch.cuda.get_device_name(device)}")


class ShallowConvNet(nn.Module):
    """
    From https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    Original paper: https://onlinelibrary.wiley.com/doi/10.1002/hbm.23730
    """

    def __init__(self, in_channels, num_classes):
        super(ShallowConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=40, kernel_size=(25, 1))
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(1, 14), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(40)
        self.pool1 = nn.AvgPool2d(kernel_size=(75, 1), stride=(15, 1))
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1800, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = torch.square(x)
        x = self.pool1(x)
        x = torch.log(x)
        x = self.drop1(x)
        x = x.view(-1, x.shape[-1] * x.shape[-2] * x.shape[-3])
        x = self.fc1(x)
        x = self.softmax(x)
        return x


class SpeechDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        # look up dataset with augmentation
        out_x = x  # torch.from_numpy(x).float().to(device)
        out_y = y  # torch.from_numpy(y).float().to(device)  # float
        return out_x.unsqueeze(0), torch.max(out_y, 0)[1]


def load_data():
    d_set = 'binary'
    d_type = 'preprocessed'

    if d_set == 'p14':
        participant = np.random.randint(1, 5)
        speech_type = np.random.choice(['imagined', 'inner'])
        if d_type == 'raw':
            filepath = f'../raw_eeg_recordings_labelled/participant_0{participant}/{speech_type}/thinking_labelled.csv'
            df = pd.read_csv(filepath)
            labels = df['Label']
            data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)  # RAW
            data = data.values
        elif d_type == 'preprocessed':
            filepath = f'../data_preprocessed/participant_0{participant}/{speech_type}/preprocessed.csv'
            df = pd.read_csv(filepath)
            labels = df['Label']
            data = df.drop(labels=['Epoch', 'Label'], axis=1)  # PREPROCESSED
            data = data.values
        elif d_type == 'time_features':
            data = np.load(f'features/even_windows/participant_0{participant}/{speech_type}/linear_features.npy')
            labels = np.load(f'features/even_windows/participant_0{participant}/{speech_type}/linear_labels.npy')
        elif d_type == 'frequency_features':
            data = np.load(f'features/even_windows/participant_0{participant}/{speech_type}/features.npy')
            labels = np.load(f'features/even_windows/participant_0{participant}/{speech_type}/labels.npy')
        elif d_type == 'mfccs':
            data = np.load(f'features/even_windows/participant_0{participant}/{speech_type}/mfcc_features.npy')
            labels = np.load(f'features/even_windows/participant_0{participant}/{speech_type}/mfcc_labels.npy')

    elif d_set == 'feis':
        if d_type == 'raw':
            filepath = f'feis_data/feis-01-thinking.csv'
            df = pd.read_csv(filepath)
            labels = df['Label']
            data = df.drop(labels=['Time:256Hz', 'Epoch', 'Label', 'Stage', 'Flag'], axis=1)  # RAW
            data = data.values
        elif d_type == 'preprocessed':
            filepath = f'feis_data/preprocessed.csv'
            df = pd.read_csv(filepath)
            labels = df['Label']
            data = df.drop(labels=['Epoch', 'Label'], axis=1)  # PREPROCESSED
            data = data.values
        elif d_type == 'time_features':
            data = np.load(f'features/even_windows/feis/linear_features.npy')
            labels = np.load(f'features/even_windows/feis/linear_labels.npy')
        elif d_type == 'frequency_features':
            data = np.load(f'features/even_windows/feis/features.npy')
            labels = np.load(f'features/even_windows/feis/labels.npy')
        elif d_type == 'mfccs':
            data = np.load(f'features/even_windows/feis/mfcc_features.npy')
            labels = np.load(f'features/even_windows/feis/mfcc_labels.npy')

    elif d_set == 'p00':
        speech_type = np.random.choice(['imagined', 'inner'])
        if d_type == 'raw':
            filepath = f'../data_preprocessed/participant_00/{speech_type}/preprocessed.csv'
            df = pd.read_csv(filepath)
            labels = df['Label']
            data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)  # RAW
            data = data.values

    elif d_set == 'binary':
        if d_type == 'raw':
            filepath = f'binary_data/p01_imagined_raw_binary.csv'
            df = pd.read_csv(filepath)
            labels = df['Label']
            data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)  # RAW
            data = data.values
        elif d_type == 'preprocessed':
            filepath = f'binary_data/p01_imagined_preprocessed_binary.csv'
            df = pd.read_csv(filepath)
            labels = df['Label']
            data = df.drop(labels=['Epoch', 'Label'], axis=1)  # PREPROCESSED
            data = data.values
        elif d_type == 'time_features':
            data = np.load(f'features/even_windows/binary/linear_features.npy')
            labels = np.load(f'features/even_windows/binary/linear_labels.npy')
        elif d_type == 'frequency_features':
            data = np.load(f'features/even_windows/binary/features.npy')
            labels = np.load(f'features/even_windows/binary/labels.npy')
        elif d_type == 'mfccs':
            data = np.load(f'features/even_windows/binary/mfcc_features.npy')
            labels = np.load(f'features/even_windows/binary/mfcc_labels.npy')

    return data, labels, d_type, d_set


def build_dataset(batch_size):
    """Encodes labels, normalises data. Returns data loaders. Splits into 90/10 train/test. Returns loaders."""
    data, labels, data_type, dataset_type = load_data()
    encoder = LabelBinarizer()
    y = encoder.fit_transform(labels)

    y = torch.from_numpy(y).float().to(device)
    data = torch.from_numpy(data).float().to(device)

    data_mean = torch.mean(data, dim=0)
    data_var = torch.var(data, dim=0)
    data_norm = (data - data_mean) / torch.sqrt(data_var)

    if dataset_type == 'p14' or dataset_type == 'p00':
        if data_type == 'raw':
            y = y.reshape(320, -1, 16)[:, 0, :]
            data_norm = data_norm.reshape(320, -1, 14)
        elif data_type == 'preprocessed':
            y = y.reshape(319, -1, 16)[:, 0, :]
            data_norm = data_norm.reshape(319, -1, 14)
        elif data_type == 'mfccs' or data_type == 'time_features' or data_type == 'frequency_features':
            data_norm = data_norm.reshape(1595, -1, 14)
    elif dataset_type == 'binary':
        if data_type == 'raw':
            y = y.reshape(40, -1, 2)[:, 0, :]
            data_norm = data_norm.reshape(40, -1, 14)
        elif data_type == 'preprocessed':
            y = y[0:30720, :]
            data_norm = data_norm[0:30720, :]
            y = y.reshape(40, -1, 2)[:, 0, :]
            data_norm = data_norm.reshape(40, -1, 14)
        elif data_type == 'mfccs' or data_type == 'time_features' or data_type == 'frequency_features':
            data_norm = data_norm.reshape(200, -1, 14)
    elif dataset_type == 'feis':
        if data_type == 'raw':
            y = y.reshape(160, -1, 16)[:, 0, :]
            data_norm = data_norm.reshape(160, -1, 14)
        elif data_type == 'preprocessed':
            y = y.reshape(159, -1, 16)[:, 0, :]
            data_norm = data_norm.reshape(159, -1, 14)
        elif data_type == 'mfccs' or data_type == 'time_features' or data_type == 'frequency_features':
            data_norm = data_norm.reshape(795, -1, 14)

    # split into 80/10 train/val
    input_train, input_test, target_train, target_test = train_test_split(data_norm, y, test_size=0.1)

    #print(f"input_train: {input_train.shape}, input_test: {input_test.shape}\n"
    #      f"target_train: {target_train.shape}, target_test: {target_test.shape}")

    train = SpeechDataset(input_train, target_train)
    test = SpeechDataset(input_test, target_test)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, data_type, dataset_type


def build_optimizer(model, optimizer, lr, wd):
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optimizer


def build_criterion(criterion):
    if criterion == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif criterion == 'NLLLoss':
        criterion = nn.NLLLoss()
    return criterion


def train_epoch(model, train_loader, val_loader, optimizer, criterion):
    running_train_loss = 0
    running_val_loss = 0
    total = 0
    running_accuracy = 0.0
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()

        loss = criterion(model(data), target)
        running_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        wandb.log({"train batch loss": loss.item()})

    train_loss_value = running_train_loss / len(train_loader)

    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            predicted_outputs = model(x)
            val_loss = criterion(predicted_outputs, y)

            _, predicted = torch.max(predicted_outputs.data, 1)
            running_val_loss += val_loss.item()
            total += y.size(0)
            wandb.log({"valid batch loss": val_loss.item()})
            running_accuracy += (predicted == y).sum().item()

    val_loss_value = running_val_loss / len(val_loader)

    val_accuracy = (100 * running_accuracy / total)

    return val_accuracy


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        train_loader, test_loader, data_type, dataset_type = build_dataset(config.batch_size)
        if dataset_type == 'binary':
            model = EEGNet(in_channels=1,
                           num_classes=2,
                           dropout_rate=config.dropout_rate,
                           data_type=data_type,
                           dataset_type=dataset_type).to(device)
        else:
            model = EEGNet(in_channels=1,
                           num_classes=16,
                           dropout_rate=config.dropout_rate,
                           data_type=data_type,
                           dataset_type=dataset_type).to(device)
        #model = ShallowConvNet(in_channels=1, num_classes=16).to(device)
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
        criterion = build_criterion(config.criterion)

        for epoch in range(config.epochs):
            metric_output = train_epoch(model, train_loader, test_loader, optimizer, criterion)
            #wandb.log({'validation_loss': avg_loss, 'epoch': epoch})
            wandb.log({'val_accuracy': metric_output, 'epoch': epoch})


if __name__ == '__main__':
    wandb.login()
    sweep_config = {'method': 'bayes'}
    #metric = {'name': 'validation_loss', 'goal': 'minimize'}
    metric = {'name': 'val_accuracy', 'goal': 'maximize'}
    sweep_config['metric'] = metric

    parameters_dict = {
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        'learning_rate': {
            'values': [0.1, 0.01, 0.001, 0.0001, 0.00001]
        },
        'batch_size': {
            'values': [#2,
                       #4,
                       #8,
                       16,
                       32,
                       64,
                       128]
        },
        'epochs': {
            'values': [10, 20, 30, 40, 50]
        },
        'weight_decay': {
            'values': [10, 1, 0, 0.1, 0.01]
        },
        'criterion': {
            'values': ['CrossEntropyLoss', 'NLLLoss']
        },
        'dropout_rate': {
            'values': [0, 0.25, 0.5, 0.75]
        }
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project='EEGNet')

    wandb.agent(sweep_id, train, count=100)

    #run_algorithm()
