import torch


if __name__ == '__main__':
    filepath = '../data_preprocessed/participant_01/imagined/preprocessed-epo.fif'

    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'

    print(device)

