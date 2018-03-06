from mlkit.torch_kit.base import BaseModel
import torch
import torch.nn.functional as F
import torch.nn as nn

from mlkit.torch_kit import utils as tu

# DEFINE YOUR NEURAL NETWORK MODELS HERE
class LinearModel(BaseModel):
    def __init__(self, n_features=1, n_outputs=10):
        super(LinearModel, self).__init__(problem_type="classification", 
                                          loss_name="categorical_crossentropy")
        self.n_outputs = n_outputs
        self.fc1 = nn.Linear(n_features, 50)
        self.fc2 = nn.Linear(50, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x)

class SmallCNNModel(BaseModel):
    def __init__(self, n_channels=1, img_dim=28, n_outputs=10):
        super(SmallCNNModel, self).__init__(problem_type="classification", 
                                       loss_name="categorical_crossentropy")
        self.n_outputs = n_outputs
        self.conv1 = nn.Conv2d(n_channels, 10, (3,3), padding=1)
        self.conv2 = nn.Conv2d(10, 20, (3,3), padding=1)

        self.flatten_size = 20*img_dim/2*img_dim/2
        self.fc1 = nn.Linear(self.flatten_size, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))

        x = x.view((-1, self.flatten_size))
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 

        return F.log_softmax(x)

class CNNModel(BaseModel):
    def __init__(self, n_channels=1, img_dim=28, n_outputs=10):
        super(CNNModel, self).__init__(problem_type="classification", 
                                       loss_name="categorical_crossentropy")
        self.n_outputs = n_outputs
        self.conv1 = nn.Conv2d(n_channels, 10, (3,3), padding=1)
        self.conv2 = nn.Conv2d(10, 10, (3,3), padding=1)

        self.flatten_size = 10*img_dim/4*img_dim/4
        self.fc1 = nn.Linear(self.flatten_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))

        x = x.view((-1, self.flatten_size))
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 

        return F.log_softmax(x)

class AttentionModel(BaseModel):
    def __init__(self, n_channels=1, n_outputs=1):
        super(AttentionModel, self).__init__(problem_type="classification", 
                                             loss_name="binary_crossentropy")

        self.conv1 = nn.Conv2d(n_channels, 10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(10, 1, kernel_size=3, padding=1)
        self.n_outputs = n_outputs


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.max_pool2d(self.conv3(x), kernel_size=2)
        x = F.sigmoid(x)


        x = F.max_pool2d(x, kernel_size=x.size()[2:])

        return torch.squeeze(x)

    def get_heatmap(self, x, output=1):
        n, _, n_rows, n_cols = x.shape

        x = tu.numpy2var(x)

        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.max_pool2d(self.conv3(x), kernel_size=2)
        x = F.sigmoid(x)

        x = tu.get_numpy(x)

        return x