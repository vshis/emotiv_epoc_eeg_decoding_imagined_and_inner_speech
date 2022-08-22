import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm
from sklearn import metrics
from torchinfo import summary
import os
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {torch.cuda.get_device_name(device)}")

NUMBER_OF_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0001


class EEGNet(nn.Module):
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
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.F1, kernel_size=(self.kernel_length, 1), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1)

        # second layer - depthwise convolution
        self.depth_conv1 = nn.Conv2d(in_channels=self.F1, out_channels=self.F2, kernel_size=(1, self.eeg_channels), groups=self.F1, bias=False)
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


class Conv2DClassifier(nn.Module):
    def __init__(self, channels_in, num_classes):
        super(Conv2DClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=12, kernel_size=(154, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(11, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(11, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(11, 3), stride=1, padding=1)

        self.fc1 = nn.Linear(48384, num_classes)

        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        #print(x.shape)
        x = self.maxpool(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        #print(x.shape)
        #print(x.shape)
        #print(x.shape)
        x = x.view(-1, x.shape[-1] * x.shape[-2] * x.shape[-3])
        x = self.fc1(x)
        #print(x.shape)
        #print(x.shape)
        #print(x.shape)
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


class Conv1DRawClassifier(nn.Module):
    def __init__(self, channels_in, num_classes):
        super(Conv1DRawClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels_in, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(168, num_classes)

        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(24)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = x.view(-1, x.shape[-1] * x.shape[-2])
        x = self.fc1(x)
        return x


class Conv1DFeaturesClassifier(nn.Module):
    def __init__(self, channels_in, num_classes):
        super(Conv1DFeaturesClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels_in, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(1176, num_classes)

        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(24)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = x.view(-1, x.shape[-1] * x.shape[-2])
        x = self.fc1(x)
        return x


class SpeechDataset(Dataset):
    def __init__(self, x, y, mode='train'):
        self.x = x
        self.y = y
        self.mode = mode

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        # look up dataset with augmentation
        out_x = x  # torch.from_numpy(x).float().to(device)
        out_y = y  # torch.from_numpy(y).float().to(device)  # float
        return out_x.unsqueeze(0), torch.max(out_y, 0)[1]


def prep_loaders(data, labels, two_d=False):
    """Encodes labels, normalises data. Returns data loaders."""
    encoder = LabelBinarizer()
    y = encoder.fit_transform(labels)

    y = torch.from_numpy(y).float().to(device)
    data = torch.from_numpy(data).float().to(device)

    data_mean = torch.mean(data, dim=0)
    data_var = torch.var(data, dim=0)
    data_norm = (data - data_mean) / torch.sqrt(data_var)

    if two_d:
        # for 2D clf
        y = y.reshape(320, -1, 16)[:, 0, :]
        data_norm = data_norm.reshape(320, -1, 14)

    # split into 80/10/10 train/val/test
    input_train, x_rem, target_train, y_rem = train_test_split(data_norm, y, test_size=0.2, random_state=42)
    input_val, input_test, target_val, target_test = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)

    print(f"input_train: {input_train.shape}, input_val: {input_val.shape}, input_test: {input_test.shape}\n"
          f"target_train: {target_train.shape}, target_val: {target_val.shape}, target_test: {target_test.shape}")
    batch_size = BATCH_SIZE

    train = SpeechDataset(input_train, target_train, mode='train')
    val = SpeechDataset(input_val, target_val, mode='val')
    test = SpeechDataset(input_test, target_test, mode='test')

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(train_loader, val_loader):
    # Hyperparameters
    n_epochs = NUMBER_OF_EPOCHS
    lr = LEARNING_RATE

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()

    # Build model, initial weight and optimizer
    #model = Conv1DRawClassifier(channels_in=1, num_classes=16).to(device)
    #model = Conv1DFeaturesClassifier(channels_in=1, num_classes=16).to(device)

    #model = Conv2DClassifier(channels_in=1, num_classes=16).to(device)

    model = EEGNet(in_channels=1, num_classes=16).to(device)
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.1)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.1)  # 6.25% test acc
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.1, momentum=0.5)  # 12.5% test acc
    #optimizer = torch.optim.ASGD(model.parameters(), lr=lr, weight_decay=0.1)  # 3.125% test acc
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    best_accuracy = 0.0

    for epoch in range(n_epochs):
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        total = 0
        #p_bar = tqdm(train_loader)
        p_bar = train_loader
        for x, y in p_bar:
            optimizer.zero_grad()  # clear gradients
            predicted_outputs = model(x)  # forward pass
            train_loss = criterion(predicted_outputs, y)  # find loss
            train_loss.backward()  # calculate gradients
            optimizer.step()  # update weights
            running_train_loss += train_loss.item()  # calculate loss

        train_loss_value = running_train_loss/len(train_loader)

        with torch.no_grad():
            model.eval()
            for x, y in val_loader:
                predicted_outputs = model(x)
                val_loss = criterion(predicted_outputs, y)

                _, predicted = torch.max(predicted_outputs.data, 1)
                running_val_loss += val_loss.item()
                total += y.size(0)
                running_accuracy += (predicted == y).sum().item()

        val_loss_value = running_val_loss/len(val_loader)

        #scheduler.step(val_loss_value)

        accuracy = (100 * running_accuracy / total)

        if accuracy > best_accuracy:
            torch.save(model.state_dict(), 'model.pt')
            best_accuracy = accuracy

        print(f"Epoch {epoch+1} \t Learning Rate: {optimizer.param_groups[0]['lr']} \t Training Loss: {train_loss_value:.4f} \t Validation Loss: {val_loss_value:.4f} \t Validation accuracy: {accuracy:.3f}%")


def test_model(test_loader):
    """Returns test accuracy"""
    #model = Conv1DRawClassifier(channels_in=1, num_classes=16).to(device)  # RAW
    #model = Conv1DFeaturesClassifier(channels_in=1, num_classes=16).to(device)  # FEATURES
    #model = Conv2DClassifier(channels_in=1, num_classes=16).to(device)
    model = EEGNet(in_channels=1, num_classes=16).to(device)
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


def run_algorithm():
    participants = [i for i in range(1, 5)]
    test_accs = {}

    for participant_n in participants:
        print(f"------Participant number {participant_n}------")
        speech_modes = ['imagined', 'inner']
        for speech_mode in speech_modes:
            print(f"-->\nSpeech mode: {speech_mode}\n")
            # FEATURES
            #data = np.load('features/even_windows/participant_01/imagined/features.npy')
            #labels = np.load('features/even_windows/participant_01/imagined/labels.npy')

            # RAW
            filepath = f'../raw_eeg_recordings_labelled/participant_0{participant_n}/{speech_mode}/thinking_labelled.csv'

            # PREPROCESSED
            #filepath = f'../data_preprocessed/participant_0{participant_n}/{speech_mode}/preprocessed.csv'

            df = pd.read_csv(filepath)
            labels = df['Label']
            data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)  # RAW
            #data = df.drop(labels=['Epoch', 'Label'], axis=1)  # PREPROCESSED
            data = data.values
            train_loader, val_loader, test_loader = prep_loaders(data, labels, two_d=True)

            train_model(train_loader, val_loader)
            test_acc = test_model(test_loader)
            test_accs[f'p{participant_n}_{speech_mode}'] = test_acc
            exit()

    df = pd.DataFrame()
    for header, values in list(test_accs.items()):
        if type(values) is float:
            values = [values]
        df[header] = values

    savedir = f'classification_results/eegnet'

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    df.to_csv(Path(f'{savedir}/raw_results.csv'), index=False)


if __name__ == '__main__':

    run_algorithm()
