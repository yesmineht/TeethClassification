import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

from typing import Optional, Callable, List
#*********************************************
import os
import os.path as os
import glob
from PIL import Image
#*******************************************
import json
import cv2
# from google.colab.patches import cv2_imshow
#******************************************
# pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0cu102.html
import torch
from torch_geometric.data import Data
from torch.autograd import Variable
import torch_geometric.transforms as T
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torchvision import transforms
from torch_geometric.nn import (SplineConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)
from torch_geometric.nn import GCNConv

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import os
import json
import numpy as np
import cv2

# scaler = transforms.Scale((224, 224))
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# to_tensor = transforms.ToTensor()

# Load the pretrained model
# modell = models.resnet18(pretrained=False)
# Use the model object to select the desired layer
# layer = modell._modules.get('avgpool')
class Resnet_GNN(torch.nn.Module):
    def __init__(self, num_classes, device):
        super(Resnet_GNN, self).__init__()
        self.num_classes = num_classes
        self.device = device
        # spline x,edge_index,edge_features
        self.resnet = models.resnet18(pretrained=False)
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        # function defined earlier for fearture extraction output tensor zize 512)

        # self.conv1 = GCNConv(512, 128)
        # self.conv2 = GCNConv(128, dataset.num_classes)

        self.conv1 = SplineConv(512, 128, dim=2, kernel_size=5, aggr='max')
        self.conv3 = SplineConv(128, 64, dim=2, kernel_size=5, aggr='max')

        self.conv2 = SplineConv(64, self.num_classes, dim=2, kernel_size=5, aggr='max')

        # self.fc2 = torch.nn.Linear(512, num_classes)

        # self.fc1 = torch.nn.Linear(512, 128)
        # self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x, edge_index, edge_attr):
        # X1 pour le training ta3 resnet
        # x1=self.resnet(x)
        x = self.feature_extractor(x).squeeze()

        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x

    # def get_vector(self, tensor_batch):
    #     t_img = tensor_batch.to(device=self.device)
    #     x = self.feature_extractor(t_img).squeeze()
    #     return x



        # for x in tensor_batch:
        #     x.to(device=self.device)
        #
        #     np_arr = x.cpu().detach().numpy()
        #
        #     image = np.transpose(np_arr, (2, 0, 1)).astype(np.float32)
        #
        #     x_after_transpose = torch.tensor(image, dtype=torch.float)
        #     t_img = Variable(normalize(x_after_transpose).unsqueeze(0)).to(device=self.device)
        #
        #     def copy_data(m, i, o):
        #         my_embedding.copy_(o.flatten())  # <-- flatten
        #
        #     h = self.layer.register_forward_hook(copy_data)
        #     self.resnet(t_img)
        #     k.append(my_embedding.cpu().detach().numpy())
        #     h.remove()
        #     my_embedding = torch.zeros(512)
        # s = np.asarray(k)
        #
        # return torch.tensor(s, dtype=torch.float)


# model = Resnet_GNN()
# print(model)
