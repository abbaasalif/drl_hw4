import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - kernel_size) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size = 7)), kernel_size = 4)   
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size = 7)), kernel_size = 4)
        
        linear_input_size = convw * convh * 64
        self.linear_1 = nn.Sequential(nn.Linear(linear_input_size, 512), nn.ReLU(),)
        self.actor = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.actor(self.linear_1(x.view(x.size(0), -1)))
