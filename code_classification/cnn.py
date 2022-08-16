import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {torch.cuda.get_device_name(device)}")


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()

        self.conv = nn.Conv1d(in_channels=14, out_channels=32, kernel_size=5, stride=1)

        self.conv_pad = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.drop_50 = nn.Dropout(p=0.5)

        self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2)

        self.dense1 = nn.Linear(32 * 8, 32)
        self.dense2 = nn.Linear(32, 32)

        self.dense_final = nn.Linear(32, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        residual = self.conv(x)
        print()

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
        x = x.view(-1, 32 * 8)  # Reshape (current_dim, 32*2)
        x = F.relu(self.dense1(x))
        # x = self.drop_60(x)
        x = self.dense2(x)
        x = self.softmax(self.dense_final(x))
        return x


if __name__ == '__main__':
    df = pd.read_csv('../raw_eeg_recordings_labelled/participant_01/imagined/thinking_labelled.csv')
    labels = df['Label']
    data = df.drop(labels=['Epoch', 'Label', 'Stage'], axis=1)
    encoder = LabelBinarizer()
    y = encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)

    model = Classifier(input_size=1, num_classes=16).to(device)

    x_train_tensor = torch.tensor(x_train.values).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    print(x_train_tensor.shape)

    print(x_train_tensor[0].shape)
    model(x_train_tensor[0])

