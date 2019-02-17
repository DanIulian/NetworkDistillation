import torch.nn as nn
import torch.nn.functional as F


def get_model_class(model_name):

    if model_name == "MLPTan":
        return MLPTan
    if model_name == "MLPSigmoid":
        return MLPSigmoid
    if model_name == "MLPRelu":
        return MLPRelu
    if model_name == "CNNSimple":
        return CNNSimple
    if model_name == "MLPTeacher":
        return MLPTeacher
    if model_name == "MLPStudent":
        return MLPStudent
    if model_name == "CNNTeacher":
        return SimpleNet

    return None


class MLPTan(nn.Module):
    def __init__(self, input_size, output_size=10, dp=None):
        super(MLPTan, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256), nn.Tanh(),
            nn.Linear(in_features=256, out_features=128), nn.Tanh(),
            nn.Linear(in_features=128, out_features=output_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.nn(x)

    def set_eval_mode(self):
        """ Set agent to evaluation mode"""
        self.eval()

    def set_train_mode(self):
        """ Set agent to training mode """
        self.train()


class MLPRelu(nn.Module):
    def __init__(self, input_size, output_size=10, dp=None):
        super(MLPRelu, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256), nn.ReLU(),
            nn.Linear(in_features=256, out_features=128), nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.nn(x)

    def set_eval_mode(self):
        """ Set agent to evaluation mode"""
        self.eval()

    def set_train_mode(self):
        """ Set agent to training mode """
        self.train()


class MLPSigmoid(nn.Module):
    def __init__(self, input_size, output_size=10, dp=None):
        super(MLPSigmoid, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256), nn.Sigmoid(),
            nn.Linear(in_features=256, out_features=128), nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=output_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.nn(x)

    def set_eval_mode(self):
        """ Set agent to evaluation mode"""
        self.eval()

    def set_train_mode(self):
        """ Set agent to training mode """
        self.train()


class CNNSimple(nn.Module):
    def __init__(self, input_size, output_size=10, dp=None):
        super(CNNSimple, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_size,
                out_channels=64,
                kernel_size=5,
                stride=3,
                padding=0,
                bias=True), nn.ReLU())
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True), nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(in_features=8192, out_features=128), nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_size))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def set_eval_mode(self):
        """ Set agent to evaluation mode"""
        self.eval()

    def set_train_mode(self):
        """ Set agent to training mode """
        self.train()


class MLPTeacher(nn.Module):
    def __init__(self, input_size, output_size=10, dp=None):
        super(MLPTeacher, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=512), nn.ReLU(),
            nn.Dropout(dp), nn.Linear(in_features=512, out_features=512),
            nn.ReLU(), nn.Dropout(dp),
            nn.Linear(in_features=512, out_features=output_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.nn(x)

    def set_eval_mode(self):
        """ Set agent to evaluation mode"""
        self.eval()

    def set_train_mode(self):
        """ Set agent to training mode """
        self.train()


class MLPStudent(nn.Module):
    def __init__(self, input_size, output_size=10, dp=None):
        super(MLPStudent, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=64), nn.ReLU(),
            nn.Dropout(dp), nn.Linear(in_features=64, out_features=64),
            nn.ReLU(), nn.Dropout(dp),
            nn.Linear(in_features=64, out_features=output_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.nn(x)

    def set_eval_mode(self):
        """ Set agent to evaluation mode"""
        self.eval()

    def set_train_mode(self):
        """ Set agent to training mode """
        self.train()


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            kernel_size=3,
            out_channels=out_channels,
            stride=1,
            padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output


class SimpleNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, dropout=10):
        super(SimpleNet, self).__init__()

        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3, out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(
            self.unit1, self.unit2, self.unit3, self.pool1, self.unit4,
            self.unit5, self.unit6, self.unit7, self.pool2, self.unit8,
            self.unit9, self.unit10, self.unit11, self.pool3, self.unit12,
            self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 128)
        output = self.fc(output)
        return output

    def set_eval_mode(self):
        """ Set agent to evaluation mode"""
        self.eval()

    def set_train_mode(self):
        """ Set agent to training mode """
        self.train()

