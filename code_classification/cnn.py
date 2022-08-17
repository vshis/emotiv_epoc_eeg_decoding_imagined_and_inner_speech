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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {torch.cuda.get_device_name(device)}")


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()

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


class Conv1DClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Conv1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=2)
        self.dense1 = nn.Linear(512, 64)
        self.dense2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(8, -1)  # Reshape (current_dim, 32*16)
        x = self.dense1(x)
        x = torch.sigmoid(self.dense2(x))
        return x


class ShallowClassifier(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(ShallowClassifier, self).__init__()
        self.hidden = nn.Linear(input_num, hidden_num)  # hidden layer
        self.output = nn.Linear(hidden_num, output_num)  # output layer
        self.sigmoid = nn.Sigmoid()  # sigmoid activation function
        self.relu = nn.ReLU()  # relu activation function

    def forward(self, x):
        x = self.relu(self.hidden(x))
        out = self.output(x)
        return self.sigmoid(out)


class SpeechDataset(Dataset):
    def __init__(self, x_train, y_train, mode='train'):
        self.x_train = x_train
        self.y_train = y_train
        self.mode = mode

    def __len__(self):
        return len(self.x_train)

    def _augmentations(self, x_data, y_data):
        # flip
        if np.random.rand() < 0.5:
            x_data = x_data[::-1]
            y_data = y_data[::-1]
        return x_data, y_data

    def __getitem__(self, idx):
        x = self.x_train[idx]
        y = self.y_train[idx]
        # look up dataset with augmentation
        # if self.mode == 'train':
        #    x, y = self._augmentations(x, y)
        out_x = torch.from_numpy(x).float().to(device)
        out_y = torch.from_numpy(y).float().to(device)  # float
        return out_x, out_y


def using_features():
    data = np.load('features/even_windows/participant_01/imagined/features.npy')
    labels = np.load('features/even_windows/participant_01/imagined/labels.npy')
    train_loader, val_loader, test_loader = prep_loaders(data, labels)
    #train_model(train_loader)
    test_model(test_loader)


def using_raw():
    df = pd.read_csv('../raw_eeg_recordings_labelled/participant_01/imagined/thinking_labelled.csv')
    labels = df['Label']
    data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)
    data = data.values
    train_loader, val_loader, test_loader = prep_loaders(data, labels)
    train_model(train_loader)


def prep_loaders(data, labels):
    encoder = LabelBinarizer()
    y = encoder.fit_transform(labels)

    # split into 80/10/10 train/val/test
    input_train, x_rem, target_train, y_rem = train_test_split(data, y, test_size=0.2, random_state=42)
    input_val, input_test, target_val, target_test = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)

    print(f"input_train: {input_train.shape}, input_val: {input_val.shape}, input_test: {input_test.shape}\n"
          f"target_train: {target_train.shape}, target_val: {target_val.shape}, target_test: {target_test.shape}")

    batch_size = 8

    train = SpeechDataset(input_train, target_train, mode='train')
    val = SpeechDataset(input_val, target_val, mode='val')
    test = SpeechDataset(input_test, target_test, mode='test')

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(train_loader):
    # Hyperparameters
    n_epochs = 10
    lr = 0.001

    loss_fn = nn.MSELoss()

    # Build model, initial weight and optimizer
    #model = Conv1DClassifier(input_size=1, num_classes=16).to(device)
    model = ShallowClassifier(input_num=98, hidden_num=64, output_num=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Using Adam optimizer
    loss_his, train_loss = [], []
    model.train()

    for epoch in range(n_epochs):
        p_bar = tqdm(train_loader)
        for i, (x, y) in enumerate(p_bar):
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            p_bar.set_description(f"[Loss: {train_loss[-1]}")
            if i % 50 == 0:
                loss_his.append(np.mean(train_loss))
                train_loss.clear()
        print(f"Epoch {epoch + 1}/{n_epochs} [Loss: {loss_his[-1]}")

    torch.save(model.state_dict(), 'model.pt')


def test_model(testloader):
    model = ShallowClassifier(input_num=98, hidden_num=64, output_num=16).to(device)
    model.load_state_dict(torch.load('model.pt'))
    y_pred = []
    y_true = []
    with torch.no_grad():
        for x, y in tqdm(testloader):
            x = x.to(device)
            pred = model(x).squeeze(dim=-1).detach().cpu().numpy()
            y_pred.append(pred)
            y_true.append(y.detach().cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_true[y_true < .1] = 0
    print('auc roc: ', metrics.roc_auc_score(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")


if __name__ == '__main__':
    using_features()
    #using_raw()
