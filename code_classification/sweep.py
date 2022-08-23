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
from cnn import Conv2DMFCCClassifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {torch.cuda.get_device_name(device)}")

NUMBER_OF_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.1


class EEGNet(nn.Module):
    """From https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py"""

    def __init__(self, in_channels, num_classes):
        super(EEGNet, self).__init__()

        # parameters
        self.D = 2  # depth multiplier for depth wise convolution - number of spatial filters to learn
        self.kernel_length = 128  # kernel size for first convolution, half the sampling rate
        self.eeg_channels = 14  # number of EEG channels
        self.F1 = 8  # number of temporal filters
        self.F2 = self.D * self.F1  # number of point wise filters to learn
        self.dropout_rate = 0.5  # dropout rate

        # first layer - temporal convolution
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.F1, kernel_size=(self.kernel_length, 1),
                               padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1)

        # second layer - depthwise convolution
        self.depth_conv1 = nn.Conv2d(in_channels=self.F1, out_channels=self.F2, kernel_size=(1, self.eeg_channels),
                                     groups=self.F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F2)
        self.pooling1 = nn.AvgPool2d(kernel_size=(4, 1))

        # third layer - depthwise-separable convolution (depthwise convolution + pointwise convolution)
        self.depth_conv2 = nn.Conv2d(in_channels=self.F2, out_channels=self.F2, kernel_size=(16, 1),
                                     groups=self.F2, bias=False)
        # pointwise convolution
        self.point_conv = nn.Conv2d(in_channels=self.F2, out_channels=self.F2, kernel_size=1, bias=False)
        self.separable_conv = torch.nn.Sequential(self.depth_conv2, self.point_conv)
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.pooling2 = nn.AvgPool2d(kernel_size=(8, 1))

        # fully connected output
        self.fc1 = nn.Linear(352, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        # Layer 2
        x = self.depth_conv1(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = F.dropout(x, self.dropout_rate)
        # Layer 3
        x = self.separable_conv(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout_rate)
        # Layer 4
        x = x.view(-1, x.shape[-1] * x.shape[-2] * x.shape[-3])
        x = self.fc1(x)
        x = self.softmax(x)
        return x


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


def prep_data(data, labels, two_d=False, data_type='raw'):
    """returns normalised data and labels tensors"""
    encoder = LabelBinarizer()
    y = encoder.fit_transform(labels)

    y = torch.from_numpy(y).float().to(device)
    data = torch.from_numpy(data).float().to(device)

    data_mean = torch.mean(data, dim=0)
    data_var = torch.var(data, dim=0)
    data_norm = (data - data_mean) / torch.sqrt(data_var)

    # for 2D clf
    if two_d:
        if data_type == 'raw':
            y = y.reshape(320, -1, 16)[:, 0, :]
            data_norm = data_norm.reshape(320, -1, 14)
        elif data_type == 'preprocessed':
            y = y.reshape(319, -1, 16)[:, 0, :]
            data_norm = data_norm.reshape(319, -1, 14)

    return data_norm, y


def build_dataset(batch_size, data_type='raw'):
    """Encodes labels, normalises data. Returns data loaders. Splits into 90/10 train/test. Returns loaders."""
    #filepath = f'../raw_eeg_recordings_labelled/participant_01/imagined/thinking_labelled.csv'
    #df = pd.read_csv(filepath)
    #labels = df['Label']
    #data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)  # RAW
    #data = data.values
    data = np.load(f'features/even_windows/participant_01/imagined/mfcc_features.npy')
    labels = np.load(f'features/even_windows/participant_01/imagined/mfcc_labels.npy')

    encoder = LabelBinarizer()
    y = encoder.fit_transform(labels)

    y = torch.from_numpy(y).float().to(device)
    data = torch.from_numpy(data).float().to(device)

    data_mean = torch.mean(data, dim=0)
    data_var = torch.var(data, dim=0)
    data_norm = (data - data_mean) / torch.sqrt(data_var)

    if data_type == 'raw':
        y = y.reshape(320, -1, 16)[:, 0, :]
        data_norm = data_norm.reshape(320, -1, 14)
    elif data_type == 'preprocessed':
        y = y.reshape(319, -1, 16)[:, 0, :]
        data_norm = data_norm.reshape(319, -1, 14)
    elif data_type == 'mfccs' or data_type == 'time_features' or data_type == 'frequency_features':
        data_norm = data_norm.reshape(1595, -1, 14)
    # split into 80/10 train/val
    input_train, input_test, target_train, target_test = train_test_split(data_norm, y, test_size=0.1)

    #print(f"input_train: {input_train.shape}, input_test: {input_test.shape}\n"
    #      f"target_train: {target_train.shape}, target_test: {target_test.shape}")

    train = SpeechDataset(input_train, target_train)
    test = SpeechDataset(input_test, target_test)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


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

        train_loader, test_loader = build_dataset(config.batch_size, data_type='mfccs')
        #model = EEGNet(in_channels=1, num_classes=16).to(device)
        #model = ShallowConvNet(in_channels=1, num_classes=16).to(device)
        model = Conv2DMFCCClassifier(channels_in=1, num_classes=16).to(device)
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
        criterion = build_criterion(config.criterion)

        for epoch in range(config.epochs):
            #avg_loss = train_epoch(model, train_loader, test_loader, optimizer, criterion)
            metric_output = train_epoch(model, train_loader, test_loader, optimizer, criterion)
            #wandb.log({'validation_loss': avg_loss, 'epoch': epoch})
            wandb.log({'val_accuracy': metric_output, 'epoch': epoch})


def run_algorithm():
    participants = [i for i in range(1, 5)]

    for participant_n in participants:
        results = {}
        print(f"------\nParticipant number {participant_n}\n------")
        data_types = ['raw', 'preprocessed']
        for data_type in data_types:
            print(f"----Data type: {data_type}----")
            speech_modes = ['imagined', 'inner']
            for speech_mode in speech_modes:
                print(f"-->\nSpeech mode: {speech_mode}\n")
                # FEATURES
                # data = np.load('features/even_windows/participant_01/imagined/features.npy')
                # labels = np.load('features/even_windows/participant_01/imagined/labels.npy')

                # RAW
                if data_type == 'raw':
                    filepath = f'../raw_eeg_recordings_labelled/participant_0{participant_n}/{speech_mode}/thinking_labelled.csv'
                    df = pd.read_csv(filepath)
                    labels = df['Label']
                    data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)  # RAW

                # PREPROCESSED
                elif data_type == 'preprocessed':
                    filepath = f'../data_preprocessed/participant_0{participant_n}/{speech_mode}/preprocessed.csv'
                    df = pd.read_csv(filepath)
                    labels = df['Label']
                    data = df.drop(labels=['Epoch', 'Label'], axis=1)  # PREPROCESSED

                data = data.values
                # train_loader, val_loader, test_loader = prep_data(data, labels, two_d=True)
                data, labels = prep_data(data, labels, two_d=True, data_type=data_type)
                dataset = SpeechDataset(data, labels)

                folds = 5
                kfold = KFold(n_splits=folds, shuffle=True)
                accs = []
                for fold_n, (train_ids, val_test_ids) in enumerate(kfold.split(dataset)):
                    print(f"--------------------------\nFold {fold_n + 1}/{folds}\n--------------------------")
                    val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5)

                    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
                    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

                    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
                    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_subsampler)
                    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_subsampler)

                    train_model(train_loader, val_loader)
                    test_acc = test_model(test_loader)
                    accs.append(test_acc)

                mean = float(np.mean(accs))
                std = float(np.std(accs))
                print(f"Participant 0{participant_n} {speech_mode} speech mean (std) accuracy = {mean:.2f} ({std:.2f})")
                results[f'mean_p{participant_n}_{speech_mode}'] = mean
                results[f'std_p{participant_n}_{speech_mode}'] = std

        df = pd.DataFrame()
        for header, values in list(results.items()):
            if type(values) is float:
                values = [values]
            df[header] = values

        savedir = f'classification_results/shallow_conv_net'

        if not os.path.exists(savedir):
            os.makedirs(savedir)
        df.to_csv(Path(f'{savedir}/p0{participant_n}_results.csv'), index=False)


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
        }
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project='EEGNet')

    wandb.agent(sweep_id, train, count=100)

    #run_algorithm()
