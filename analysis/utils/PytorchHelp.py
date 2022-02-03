# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


"""
# help class for torch data loading
class ClassifierDataset(data.Dataset):
    def __init__(self, epoch, steps_per_epoch, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

        self.epoch = epoch
        self.iter = steps_per_epoch

    def __len__(self):
        #return len(self.X_data)
        return self.iter

    def __getitem__(self, idx):

        new_idx = idx + (self.iter*self.epoch)

         if new_idx >= len(self.X_data):
            new_idx = new_idx % len(self.X_data)

        return self.X_data[new_idx], self.y_data[new_idx]
"""


class ClassifierDataset(data.Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


# torch Multiclassifer
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class, means, stds):
        super(MulticlassClassification, self).__init__()

        # custom normalisation layer
        self.norm = NormalizeInputs(means, stds)

        self.layer_1 = nn.Linear(num_feature, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.norm(x)
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


class NormalizeInputs(nn.Module):
    def __init__(self, means, stds):
        super(NormalizeInputs, self).__init__()
        self.mean = torch.tensor(means)
        self.std = torch.tensor(stds)

    def forward(self, input):
        x = input - self.mean
        x = x / self.std
        return x
