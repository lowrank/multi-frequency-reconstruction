import sys
import os

import numpy as np

import scipy.io
from matplotlib import pyplot as plt

import matplotlib

import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader, random_split
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

    
from torch.utils.data.dataset import Dataset

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=False),
            nn.LeakyReLU(negative_slope=0.001,inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=False),
            nn.LeakyReLU(negative_slope=0.001,inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=(2,2))

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class u_net(nn.Module):
    def __init__(self, in_channels, feat):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels,   feat)
        self.down_convolution_2 = DownSample(    feat,  2 * feat)
        self.down_convolution_3 = DownSample(2 * feat,  4 * feat)
        self.down_convolution_4 = DownSample(4 * feat,  8 * feat)

        self.bottle_neck        = DoubleConv(8 * feat, 16 * feat)

        self.up_convolution_1 = UpSample(16 * feat + 8 * feat,  8 * feat)
        self.up_convolution_2 = UpSample(8  * feat + 4 * feat,  4 * feat)
        self.up_convolution_3 = UpSample(4  * feat + 2 * feat,  2 * feat)
        self.up_convolution_4 = UpSample(2  * feat +     feat,      feat)

        self.out = nn.Conv2d(in_channels=feat, out_channels=1, kernel_size=1, bias=False)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
       
        return out


class MF_Dataset(Dataset):
    def __init__(self, xdata, ydata):
        self.xdata = torch.tensor(xdata).unsqueeze(dim=1)
        self.ydata = torch.tensor(ydata).unsqueeze(dim=1)
        
    def __getitem__(self, index):
        x_img = self.xdata[index]
        y_img = self.ydata[index]

        return x_img, y_img

    def __len__(self):
        return len(self.xdata)
    
# methods    
def get_data(out_path, train_folder_paths, data_folder_path):
    
    out_folder_path = out_path
    
    if len(train_folder_paths) == 0:
        exit('Not training path included.')
    
    x_folder_path   = train_folder_paths[0] # all training folder should have the same formats.
    y_folder_path   = data_folder_path

    # List all .mat files in the folder
    x_mat_files = sorted([f for f in os.listdir(x_folder_path) if f.endswith('.mat')])
    y_mat_files = sorted([f for f in os.listdir(y_folder_path) if f.endswith('.mat')])

    x_train_list = []  # list of artifact images with missing angle (180 - ANGLE)
    y_train_list = []  # list of true images
    sample_size = len(x_mat_files)
    train_size = int(sample_size * 4/5)

    for i in range(sample_size):
        packed_data = []
        
        for x_folder_path in train_folder_paths:
            x_mat = scipy.io.loadmat(os.path.join(x_folder_path, x_mat_files[i]))
            x_train = x_mat['value']
            packed_data.append(x_train)

        if out_folder_path:
            out_mat = scipy.io.loadmat(os.path.join(out_folder_path, x_mat_files[i]))
            out_train = out_mat['value']
            packed_data.append(out_train)
            x_train_list.append(packed_data)
        else:
            x_train_list.append(packed_data)

        y_mat = scipy.io.loadmat(os.path.join(y_folder_path, y_mat_files[i]))
        y_train = y_mat['value']
        y_train_list.append(y_train)

    x_train_full = np.array(x_train_list)
    y_train_full = np.array(y_train_list)

    # training data
    xdata = x_train_full[:train_size]
    ydata = y_train_full[:train_size]

    test_xdata = x_train_full[train_size::]
    test_ydata = y_train_full[train_size::]
    
    
    return xdata, ydata, test_xdata, test_ydata

def gen_data(output_folder, train_folders, out_folder, data_folder, model):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    out_folder_path = out_folder

    output_folder = output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    x_folder_path = train_folders[0] # all folders should have the same format.
    y_folder_path = data_folder

    # List all .mat files in the folder
    x_mat_files = sorted([f for f in os.listdir(x_folder_path) if f.endswith('.mat')])

    for i in range(len(x_mat_files)):
        packed_data = []
        for x_folder_path in train_folders:
            x_mat = scipy.io.loadmat(os.path.join(x_folder_path, x_mat_files[i]))
            x_train = x_mat['value']
            packed_data.append(x_train)
        
        if out_folder_path:
            out_mat = scipy.io.loadmat(os.path.join(out_folder_path, x_mat_files[i]))
            out_train = out_mat['value']
            packed_data.append(out_train)
            out_mat = model(torch.tensor(np.array(packed_data)).unsqueeze(dim=0).float().to(device)).detach().cpu().numpy()[0][0]
        else:
            out_mat = model(torch.tensor(np.array(packed_data)).unsqueeze(dim=0).float().to(device)).detach().cpu().numpy()[0][0]


        scipy.io.savemat(os.path.join(output_folder, x_mat_files[i]), {'value': out_mat})