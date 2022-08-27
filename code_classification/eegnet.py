"""
References:
    Vernon J Lawhern et al. “EEGNet: a compact convolutional neural network for EEG-based
    brain-computer interfaces”. en. In: J Neural Eng 15.5 (June 2018), p. 056013.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from pathlib import Path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {torch.cuda.get_device_name(device)}")


# HYPERPARAMETERS
BATCH_SIZE = 64
# CRITERION = nn.CrossEntropyLoss()
CRITERION = nn.NLLLoss()
DROPOUT_RATE = 0
NUMBER_OF_EPOCHS = 50
LEARNING_RATE = 0.01
# OPTIMIZER = 'ADAM'
OPTIMIZER = 'SGD'
WEIGHT_DECAY = 0.01


class EEGNet(nn.Module):
    """
    Implementation of EEGNet in PyTorch, originally written by the authors in TensorFlow, reference:
    Vernon J Lawhern et al. “EEGNet: a compact convolutional neural network for EEG-based
    brain-computer interfaces”. en. In: J Neural Eng 15.5 (June 2018), p. 056013.
    """

    def __init__(self, in_channels, num_classes, data_type='raw', dataset_type='p14', dropout_rate=0.5):
        super(EEGNet, self).__init__()

        # parameters
        if data_type == 'raw' or data_type == 'preprocessed':
            self.D = 2  # depth multiplier for depth wise convolution - number of spatial filters to learn
            self.conv1_kernel_size = (128, 1)  # kernel size for first convolution, half the sampling rate
            self.F1 = 8  # number of temporal filters
            self.fc_size = 352  # size of the first fully connected layer
            self.pool1_size = (4, 1)
            self.pool2_size = (8, 1)
            self.depth_conv2_kernel_size = (16, 1)
        elif data_type == 'time_features' or data_type == 'mfccs':
            self.D = 2  # depth multiplier for depth wise convolution - number of spatial filters to learn
            self.conv1_kernel_size = (3, 1)  # kernel size for first convolution, half the sampling rate
            self.F1 = 8  # number of temporal filters
            self.fc_size = 16  # size of the first fully connected layer
            self.pool1_size = (2, 1)
            self.pool2_size = (4, 1)
            self.depth_conv2_kernel_size = (2, 1)
        elif data_type == 'frequency_features':
            self.D = 2  # depth multiplier for depth wise convolution - number of spatial filters to learn
            self.conv1_kernel_size = (3, 1)  # kernel size for first convolution, half the sampling rate
            self.F1 = 8  # number of temporal filters
            self.fc_size = 16  # size of the first fully connected layer
            self.pool1_size = (2, 1)
            self.pool2_size = (2, 1)
            self.depth_conv2_kernel_size = (2, 1)

        if dataset_type == 'p00':
            self.conv1_kernel_size = 64
            self.fc_size = 160
        elif dataset_type == 'binary':
            if data_type == 'raw' or data_type == 'preprocessed':
                self.fc_size = 352
            else:
                self.fc_size = 16
        elif dataset_type == 'feis':
            if data_type == 'raw' or data_type == 'preprocessed':
                self.fc_size = 608
            else:
                self.fc_size = 16

        self.F2 = self.D * self.F1  # number of point wise filters to learn
        self.dropout_rate = dropout_rate  # dropout rate
        self.eeg_channels = 14  # number of EEG channels

        # first layer - temporal convolution
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.F1, kernel_size=self.conv1_kernel_size,
                               padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1)

        # second layer - depthwise convolution
        self.depth_conv1 = nn.Conv2d(in_channels=self.F1, out_channels=self.F2, kernel_size=(1, self.eeg_channels),
                                     groups=self.F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F2)
        self.pooling1 = nn.AvgPool2d(kernel_size=self.pool1_size)

        # third layer - depthwise-separable convolution (depthwise convolution + pointwise convolution)
        self.depth_conv2 = nn.Conv2d(in_channels=self.F2, out_channels=self.F2,
                                     kernel_size=self.depth_conv2_kernel_size,
                                     groups=self.F2, bias=False)
        # pointwise convolution
        self.point_conv = nn.Conv2d(in_channels=self.F2, out_channels=self.F2, kernel_size=1, bias=False)
        self.separable_conv = torch.nn.Sequential(self.depth_conv2, self.point_conv)
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.pooling2 = nn.AvgPool2d(kernel_size=self.pool2_size)

        # fully connected output
        self.fc1 = nn.Linear(self.fc_size, num_classes)
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


def prep_data(data, labels, data_type='raw', dataset_type='p14'):
    """Encodes labels, normalises data. Returns data loaders."""
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

    return data_norm, y


def train_model(train_loader, val_loader, data_type, dataset_type, num_classes=16):
    # Hyperparameters
    n_epochs = NUMBER_OF_EPOCHS
    lr = LEARNING_RATE

    criterion = CRITERION

    model = EEGNet(in_channels=1,
                   num_classes=num_classes,
                   data_type=data_type,
                   dropout_rate=DROPOUT_RATE,
                   dataset_type=dataset_type).to(device)

    # print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if OPTIMIZER == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=0.9,
                                    weight_decay=WEIGHT_DECAY)

    best_accuracy = 0.0

    for epoch in range(n_epochs):
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        total = 0
        # p_bar = tqdm(train_loader)
        p_bar = train_loader
        model.train()
        for x, y in p_bar:
            optimizer.zero_grad()  # clear gradients
            predicted_outputs = model(x)  # forward pass
            train_loss = criterion(predicted_outputs, y)  # find loss
            train_loss.backward()  # calculate gradients
            optimizer.step()  # update weights
            running_train_loss += train_loss.item()  # calculate loss

        train_loss_value = running_train_loss / len(train_loader)

        with torch.no_grad():
            model.eval()
            for x, y in val_loader:
                predicted_outputs = model(x)
                val_loss = criterion(predicted_outputs, y)

                _, predicted = torch.max(predicted_outputs.data, 1)
                running_val_loss += val_loss.item()
                total += y.size(0)
                running_accuracy += (predicted == y).sum().item()

        val_loss_value = running_val_loss / len(val_loader)

        accuracy = (100 * running_accuracy / total)

        if accuracy > best_accuracy:
            torch.save(model.state_dict(), 'model.pt')
            best_accuracy = accuracy
        # if (epoch + 1) % 10 == 0:
        #    print(
        #        f"Epoch {epoch + 1} \t Learning Rate: {optimizer.param_groups[0]['lr']} \t Training Loss: {train_loss_value:.4f} \t Validation Loss: {val_loss_value:.4f} \t Validation accuracy: {accuracy:.3f}%")


def predict_model(test_loader, data_type, dataset_type, num_classes=16):
    """Returns test accuracy"""
    model = EEGNet(in_channels=1,
                   num_classes=num_classes,
                   data_type=data_type,
                   dropout_rate=0,
                   dataset_type=dataset_type).to(device)
    model.load_state_dict(torch.load('model.pt'))
    running_accuracy = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            y = y.to(torch.float32)
            predicted_outputs = model(x)
            _, predicted = torch.max(predicted_outputs, 1)
            total += y.size(0)
            running_accuracy += (predicted == y).sum().item()

        test_accuracy = (100 * running_accuracy / total)
        print(f'--- Test accuracy: {test_accuracy:.3f}%')
    return test_accuracy


def run_algorithm_for_p1to4():
    participants = [i for i in range(1, 5)]
    dataset_type = 'p14'

    results = {}
    for participant_n in participants:

        # print(f"------\nParticipant number {participant_n}\n------")
        data_types = [
            # 'raw',
            # 'preprocessed',
            # 'time_features',
            # 'frequency_features',
            'mfccs'
        ]
        for data_type in data_types:
            # print(f"Participant number {participant_n} -- Data type: {data_type}")
            speech_modes = [
                'imagined',
                'inner'
            ]
            for speech_mode in speech_modes:
                print(
                    f"\nRUNNING: Participant number {participant_n} --> Data type: {data_type} --> Speech mode: {speech_mode}\n")
                # RAW
                if data_type == 'raw':
                    filepath = f'../raw_eeg_recordings_labelled/participant_0{participant_n}/{speech_mode}/thinking_labelled.csv'
                    df = pd.read_csv(filepath)
                    labels = df['Label']
                    data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)  # RAW
                    data = data.values

                # PREPROCESSED
                elif data_type == 'preprocessed':
                    filepath = f'../data_preprocessed/participant_0{participant_n}/{speech_mode}/preprocessed.csv'
                    df = pd.read_csv(filepath)
                    labels = df['Label']
                    data = df.drop(labels=['Epoch', 'Label'], axis=1)  # PREPROCESSED
                    data = data.values

                # TIME DOMAIN FEATURES
                elif data_type == 'time_features':
                    data = np.load(
                        f'features/even_windows/participant_0{participant_n}/{speech_mode}/linear_features.npy')
                    labels = np.load(
                        f'features/even_windows/participant_0{participant_n}/{speech_mode}/linear_labels.npy')

                # FREQUENCY DOMAIN FEATURES
                elif data_type == 'frequency_features':
                    data = np.load(f'features/even_windows/participant_0{participant_n}/{speech_mode}/features.npy')
                    labels = np.load(f'features/even_windows/participant_0{participant_n}/{speech_mode}/labels.npy')

                # MFCCS
                elif data_type == 'mfccs':
                    data = np.load(
                        f'features/even_windows/participant_0{participant_n}/{speech_mode}/mfcc_features.npy')
                    labels = np.load(
                        f'features/even_windows/participant_0{participant_n}/{speech_mode}/mfcc_labels.npy')

                # train_loader, val_loader, test_loader = prep_data(data, labels, two_d=True)
                data, labels = prep_data(data, labels, data_type=data_type, dataset_type=dataset_type)
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

                    train_model(train_loader, val_loader, data_type, dataset_type=dataset_type)
                    test_acc = predict_model(test_loader, data_type, dataset_type=dataset_type)
                    accs.append(test_acc)

                mean = float(np.mean(accs))
                std = float(np.std(accs))
                print(
                    f"RESULTS :::::::::::: Participant 0{participant_n}, {data_type} data, {speech_mode} speech mean (std) accuracy = {mean:.2f} ({std:.2f})")
                results[f'mean_p{participant_n}_{speech_mode}_{data_type}'] = mean
                results[f'std_p{participant_n}_{speech_mode}_{data_type}'] = std

    df = pd.DataFrame()
    for header, values in list(results.items()):
        if type(values) is float:
            values = [values]
        df[header] = values

    savedir = f'classification_results/eegnet'

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df.to_csv(Path(f'{savedir}/p14_{data_type}_results.csv'), index=False)


def run_algorithm_for_p00():
    results = {}
    dataset_type = 'p00'
    # print(f"------\nParticipant number {participant_n}\n------")
    data_type = 'raw'
    # print(f"Participant number {participant_n} -- Data type: {data_type}")
    speech_modes = [
        'imagined',
        'inner'
    ]
    for speech_mode in speech_modes:
        print(f"Participant number 00 --> Data type: {data_type} --> Speech mode: {speech_mode}\n")
        # RAW
        filepath = f'../data_preprocessed/participant_00/{speech_mode}/preprocessed.csv'
        df = pd.read_csv(filepath)
        labels = df['Label']
        data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)  # RAW
        data = data.values

        # train_loader, val_loader, test_loader = prep_data(data, labels, two_d=True)
        data, labels = prep_data(data, labels, data_type=data_type, dataset_type=dataset_type)
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

            train_model(train_loader, val_loader, data_type, dataset_type=dataset_type)
            test_acc = predict_model(test_loader, data_type, dataset_type=dataset_type)
            accs.append(test_acc)

        mean = float(np.mean(accs))
        std = float(np.std(accs))
        print(f"Participant 00, data {data_type}, {speech_mode} speech mean (std) accuracy = {mean:.2f} ({std:.2f})")
        results[f'mean_{speech_mode}_{data_type}'] = mean
        results[f'std_{speech_mode}_{data_type}'] = std

    df = pd.DataFrame()
    for header, values in list(results.items()):
        if type(values) is float:
            values = [values]
        df[header] = values

    savedir = f'classification_results/eegnet'

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df.to_csv(Path(f'{savedir}/p00_results.csv'), index=False)


def run_algorithm_for_binary():
    results = {}
    dataset_type = 'binary'
    # print(f"------\nParticipant number {participant_n}\n------")
    data_types = [
        # 'raw',
        # 'preprocessed',
        # 'time_features',
        # 'frequency_features',
        'mfccs'
    ]
    for data_type in data_types:
        print(f"\nBinary --> Data type: {data_type}\n")
        # RAW
        if data_type == 'raw':
            filepath = f'binary_data/p01_imagined_raw_binary.csv'
            df = pd.read_csv(filepath)
            labels = df['Label']
            data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)  # RAW
            data = data.values

        # PREPROCESSED
        elif data_type == 'preprocessed':
            filepath = f'binary_data/p01_imagined_preprocessed_binary.csv'
            df = pd.read_csv(filepath)
            labels = df['Label']
            data = df.drop(labels=['Epoch', 'Label'], axis=1)  # PREPROCESSED
            data = data.values

        # TIME DOMAIN FEATURES
        elif data_type == 'time_features':
            data = np.load(f'features/even_windows/binary/linear_features.npy')
            labels = np.load(f'features/even_windows/binary/linear_labels.npy')

        # FREQUENCY DOMAIN FEATURES
        elif data_type == 'frequency_features':
            data = np.load(f'features/even_windows/binary/features.npy')
            labels = np.load(f'features/even_windows/binary/labels.npy')

        # MFCCS
        elif data_type == 'mfccs':
            data = np.load(f'features/even_windows/binary/mfcc_features.npy')
            labels = np.load(f'features/even_windows/binary/mfcc_labels.npy')

        # train_loader, val_loader, test_loader = prep_data(data, labels, two_d=True)
        data, labels = prep_data(data, labels, data_type=data_type, dataset_type=dataset_type)
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

            train_model(train_loader, val_loader, data_type, dataset_type=dataset_type, num_classes=2)
            test_acc = predict_model(test_loader, data_type, dataset_type=dataset_type, num_classes=2)
            accs.append(test_acc)

        mean = float(np.mean(accs))
        std = float(np.std(accs))
        print(f"Binary, {data_type} data, mean (std) accuracy = {mean:.2f} ({std:.2f})")
        results[f'mean_{data_type}'] = mean
        results[f'std_{data_type}'] = std

    df = pd.DataFrame()
    for header, values in list(results.items()):
        if type(values) is float:
            values = [values]
        df[header] = values

    savedir = f'classification_results/eegnet'

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df.to_csv(Path(f'{savedir}/binary_{data_type}_results.csv'), index=False)


def run_algorithm_for_feis():
    results = {}
    dataset_type = 'feis'
    # print(f"------\nParticipant number {participant_n}\n------")
    data_types = [
        # 'raw',
        # 'preprocessed',
        # 'time_features',
        # 'frequency_features',
        'mfccs'
    ]
    for data_type in data_types:
        print(f"FEIS --> Data type: {data_type}\n")
        # RAW
        if data_type == 'raw':
            filepath = f'feis_data/feis-01-thinking.csv'
            df = pd.read_csv(filepath)
            labels = df['Label']
            data = df.drop(labels=['Time:256Hz', 'Epoch', 'Label', 'Stage', 'Flag'], axis=1)  # RAW
            data = data.values

        # PREPROCESSED
        elif data_type == 'preprocessed':
            filepath = f'feis_data/preprocessed.csv'
            df = pd.read_csv(filepath)
            labels = df['Label']
            data = df.drop(labels=['Epoch', 'Label'], axis=1)  # PREPROCESSED
            data = data.values

        # TIME DOMAIN FEATURES
        elif data_type == 'time_features':
            data = np.load(f'features/even_windows/feis/linear_features.npy')
            labels = np.load(f'features/even_windows/feis/linear_labels.npy')

        # FREQUENCY DOMAIN FEATURES
        elif data_type == 'frequency_features':
            data = np.load(f'features/even_windows/feis/features.npy')
            labels = np.load(f'features/even_windows/feis/labels.npy')

        # MFCCS
        elif data_type == 'mfccs':
            data = np.load(f'features/even_windows/feis/mfcc_features.npy')
            labels = np.load(f'features/even_windows/feis/mfcc_labels.npy')

        # train_loader, val_loader, test_loader = prep_data(data, labels, two_d=True)
        data, labels = prep_data(data, labels, data_type=data_type, dataset_type=dataset_type)
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

            train_model(train_loader, val_loader, data_type, dataset_type=dataset_type)
            test_acc = predict_model(test_loader, data_type, dataset_type=dataset_type)
            accs.append(test_acc)

        mean = float(np.mean(accs))
        std = float(np.std(accs))
        print(f"FEIS, {data_type} data, mean (std) accuracy = {mean:.2f} ({std:.2f})")
        results[f'mean_{data_type}'] = mean
        results[f'std_{data_type}'] = std

    df = pd.DataFrame()
    for header, values in list(results.items()):
        if type(values) is float:
            values = [values]
        df[header] = values

    savedir = f'classification_results/eegnet'

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df.to_csv(Path(f'{savedir}/feis_{data_type}_results.csv'), index=False)


if __name__ == '__main__':
    # UNCOMMENT APPROPRIATE LINE
    # run_algorithm_for_p1to4()
    # run_algorithm_for_p00()
    # run_algorithm_for_binary()
    # run_algorithm_for_feis()
    exit()
