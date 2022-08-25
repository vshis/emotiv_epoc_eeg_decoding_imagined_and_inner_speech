import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm
from sklearn import metrics
from torchinfo import summary
import os
from pathlib import Path
import wandb
import pprint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {torch.cuda.get_device_name(device)}")

BATCH_SIZE = 64
#CRITERION = nn.CrossEntropyLoss()
CRITERION = nn.NLLLoss()
DROPOUT_RATE = 0
NUMBER_OF_EPOCHS = 30
LEARNING_RATE = 0.001
OPTIMIZER = 'ADAM'
#OPTIMIZER = 'SGD'
WEIGHT_DECAY = 0.0


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


class EEGNet(nn.Module):
    """From https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py"""

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
        self.depth_conv2 = nn.Conv2d(in_channels=self.F2, out_channels=self.F2, kernel_size=self.depth_conv2_kernel_size,
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


class Conv2DClassifier(nn.Module):
    def __init__(self, channels_in, num_classes):
        super(Conv2DClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=12, kernel_size=(153, 1))

        self.fc1 = nn.Linear(612, num_classes)

        self.bn1 = nn.BatchNorm2d(12)

        self.maxpool = nn.MaxPool2d(kernel_size=12)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.maxpool(x)

        x = x.view(-1, x.shape[-1] * x.shape[-2] * x.shape[-3])
        x = self.fc1(x)
        x = self.softmax(x)
        return x


class ResClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ResClassifier, self).__init__()

        self.conv = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=4, stride=1)

        self.conv_pad = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.drop_50 = nn.Dropout(p=0.5)

        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2)

        self.dense1 = nn.Linear(96, 32)
        self.dense2 = nn.Linear(32, 32)

        self.dense_final = nn.Linear(32, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        residual = self.conv(x)

        # block1
        x = F.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x += residual
        x = F.relu(x)
        residual = self.maxpool(x)  # [512 32 90]

        # block2
        x = F.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x += residual
        x = F.relu(x)
        residual = self.maxpool(x)  # [512 32 43]

        # block3
        x = F.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x += residual
        x = F.relu(x)
        residual = self.maxpool(x)  # [512 32 20]

        # block4
        x = F.relu(self.conv_pad(residual))
        x = self.conv_pad(x)
        x += residual
        x = F.relu(x)
        x = self.maxpool(x)  # [512 32 8]

        # MLP
        x = x.view(-1, 96)  # Reshape (current_dim, 32*2)
        x = F.relu(self.dense1(x))
        # x = self.drop_60(x)
        x = self.dense2(x)
        x = self.softmax(self.dense_final(x))
        # x = self.dense_final(x)
        return x


class Conv2DMFCCClassifier(nn.Module):
    """From https://ieeexplore.ieee.org/document/8832223"""

    def __init__(self, channels_in, num_classes):
        super(Conv2DMFCCClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=20, kernel_size=(4, 4))
        self.batchnorm1 = nn.BatchNorm2d(20)

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(8, 8))
        self.batchnorm2 = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(120, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = torch.tanh(x)
        # x = self.batchnorm1(F.relu(x))
        # Layer 2
        x = self.conv2(x)
        x = torch.tanh(x)
        # x = self.batchnorm2(F.relu(x))
        # Layer 3 flatten
        x = x.view(-1, x.shape[-1] * x.shape[-2] * x.shape[-3])
        # Layer 4
        x = self.fc1(x)
        # Layer 5
        x = self.fc2(x)
        # Layer 6
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class Conv1DFeaturesClassifier(nn.Module):
    """Based on this paper: https://iopscience.iop.org/article/10.1088/1741-2552/ac4430#jneac4430s2"""

    def __init__(self, channels_in, num_classes):
        super(Conv1DFeaturesClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels_in, out_channels=32, kernel_size=20, padding='same')
        self.batchnorm1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=20)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.drop = nn.Dropout(0.5)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=6)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=6)

        # self.fc1 = nn.Linear(1920, 256)  # time features
        # self.fc1 = nn.Linear(1024, 256)  # freq features
        self.fc1 = nn.Linear(2368, 256)  # mfccs
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        # Layer 1
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        # Layer 2
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.drop(x)
        # Layer 3
        x = F.relu(self.conv3(x))
        # Layer 4
        x = self.pool1(x)
        # Layer 5
        x = F.relu(self.conv4(x))
        x = self.drop(x)
        # Layer 6
        x = x.view(-1, x.shape[-1] * x.shape[-2])
        # Layer 7
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        # Layer 8
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        # Layer 9
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        # Layer 10
        x = self.fc4(x)
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

    # Build model, initial weight and optimizer
    # model = Conv1DRawClassifier(channels_in=1, num_classes=16).to(device)
    # model = Conv1DFeaturesClassifier(channels_in=1, num_classes=16).to(device)
    # model = Conv2DMFCCClassifier(channels_in=1, num_classes=16).to(device)

    # model = Conv2DClassifier(channels_in=1, num_classes=16).to(device)

    model = EEGNet(in_channels=1,
                   num_classes=num_classes,
                   data_type=data_type,
                   dropout_rate=DROPOUT_RATE,
                   dataset_type=dataset_type).to(device)
    # model = ShallowConvNet(in_channels=1, num_classes=16).to(device)

    #print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if OPTIMIZER == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY)  # 12.5% test acc

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
        #if (epoch + 1) % 10 == 0:
        #    print(
        #        f"Epoch {epoch + 1} \t Learning Rate: {optimizer.param_groups[0]['lr']} \t Training Loss: {train_loss_value:.4f} \t Validation Loss: {val_loss_value:.4f} \t Validation accuracy: {accuracy:.3f}%")


def test_model(test_loader, data_type, dataset_type, num_classes=16):
    """Returns test accuracy"""
    # model = Conv1DRawClassifier(channels_in=1, num_classes=16).to(device)  # RAW
    # model = Conv1DFeaturesClassifier(channels_in=1, num_classes=16).to(device)  # FEATURES
    # model = Conv2DMFCCClassifier(channels_in=1, num_classes=16).to(device)
    # model = Conv2DClassifier(channels_in=1, num_classes=16).to(device)
    model = EEGNet(in_channels=1,
                   num_classes=num_classes,
                   data_type=data_type,
                   dropout_rate=0,
                   dataset_type=dataset_type).to(device)
    # model = ShallowConvNet(in_channels=1, num_classes=16).to(device)
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
            #'raw',
            #'preprocessed',
            #'time_features',
            #'frequency_features',
            'mfccs'
        ]
        for data_type in data_types:
            # print(f"Participant number {participant_n} -- Data type: {data_type}")
            speech_modes = [
                'imagined',
                'inner'
            ]
            for speech_mode in speech_modes:
                print(f"\nRUNNING: Participant number {participant_n} --> Data type: {data_type} --> Speech mode: {speech_mode}\n")
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
                    test_acc = test_model(test_loader, data_type, dataset_type=dataset_type)
                    accs.append(test_acc)

                mean = float(np.mean(accs))
                std = float(np.std(accs))
                print(f"RESULTS :::::::::::: Participant 0{participant_n}, {data_type} data, {speech_mode} speech mean (std) accuracy = {mean:.2f} ({std:.2f})")
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
            test_acc = test_model(test_loader, data_type, dataset_type=dataset_type)
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
        #'raw',
        #'preprocessed',
        #'time_features',
        #'frequency_features',
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
            test_acc = test_model(test_loader, data_type, dataset_type=dataset_type, num_classes=2)
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
        #'raw',
        #'preprocessed',
        #'time_features',
        #'frequency_features',
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
            test_acc = test_model(test_loader, data_type, dataset_type=dataset_type)
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
    #run_algorithm_for_p1to4()
    run_algorithm_for_p00()
    #run_algorithm_for_binary()
    #run_algorithm_for_feis()
    exit()
