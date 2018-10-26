import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32,kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64,kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128,kernel_size=5)
        #self.conv4 = nn.Conv2d(64, 128,padding=(1,1),kernel_size=3)
        #self.conv5 = nn.Conv2d(64,128,kernel_size=3)
        self.conv_drop = nn.Dropout2d(0.2)
        self.fc1 = nn.Linear(128*4*4, 1024)
        self.fc2 = nn.Linear(1024, nclasses)

    def forward(self, x):
        #print(x.shape)
        x1 = F.relu(self.conv1(x))
        #print(x1.shape)
        x2 = F.relu(F.max_pool2d(self.conv_drop(self.conv2(x1)),2))
        #print(x2.shape)
        x3 = F.relu(F.max_pool2d(self.conv_drop(self.conv3(x2)),2))
        #x3 = F.relu(self.conv3(x2))
        #print(x3.shape)
        #x4 = F.relu(F.max_pool2d(self.conv_drop(self.conv4(x3)),2))
        #print(x4.shape)
        #x5 = self.conv_drop(F.relu(self.conv5(x4)))
        #print(x5.shape)
        x_lin = x3.view(-1, 128*4*4)
        x_h = F.relu(self.fc1(x_lin))
        x_h = F.dropout(x_h,training=self.training)
        x_o = self.fc2(x_h)
        return F.log_softmax(x_o)
