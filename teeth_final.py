import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

from typing import Optional, Callable, List
from itertools import repeat
from extract_teeth import  *
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


class teeth_final(InMemoryDataset):

    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0]
        self.root = root
        self.data, self.slices = torch.load(path)

        # self.data=self.data[0]
        # self.slices=self.data[1]
        # self.slices=Dic

    @property
    def raw_file_names(self) -> str:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ['training.pt']

    def get(self, idx):
        data = self.data.__class__()
        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                cat_dim = self.data.__cat_dim__(key, item)
                if cat_dim is None:
                    cat_dim = 0
                s[cat_dim] = slice(start, end)
                data[key] = item[s]
            elif start + 1 == end:
                s = slices[start]
                data[key] = self._get_features_from_images(item[s])
            else:
                s = slice(start, end)
                data[key] = item[s]
        return data

    def download(self):
        pass
    def process(self):
        lutt = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        self._data_list = []
        subdir_list = [f.path for f in os.scandir(self.root) if f.is_dir()]
        for data_path in subdir_list:
            json_path = os.path.join(data_path, "coords.json")
            # Opening JSON file
            try:
                f = open(json_path)
            except:
                continue
            data_json = json.load(f)
            label = []
            images_path = []
            pos_list = []
            for k in data_json.keys():
                lbl = int(k)
                if lbl < 1 or lbl > 28:
                    continue
                lbl = lutt[lbl]
                if lbl < 0:
                    continue
                label.append(lbl)
                pos_list.append([data_json[k]['c_x'], data_json[k]['c_y']])
                images_path.append(os.path.join(data_path,"{}.png".format(k)))
            # images = self._get_features_from_images(images_path)
            labels = torch.tensor(label)  #self._get_labels(label)
            pos = torch.tensor(pos_list) # self._get_pos(pos_list)
            # data = Data(x=images, y=labels, pos=pos)
            data = Data(x=images_path, y=labels, pos=pos)
            self._data_list.append(data)
        # data = self.collate(data_list)
        if self.pre_filter is not None:
            self._data_list = [d for d in self._data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            self._data_list = [self.pre_transform(d) for d in self._data_list]

        torch.save(self.collate(self._data_list), self.processed_paths[0])


    def _get_features_from_images(self, list):
        images = []
        for filename in list:
            # image = Image.open(i)
            # image = image.resize((256, 256))
            input_image = Image.open(filename)
            input_image = input_image.resize((256, 256))
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image)
            # image = np.array(image)
            # image = image.ravel()
            images.append(input_tensor)
        # images = np.asarray(images)
        # return torch.tensor(images, dtype=torch.float)
        return  torch.stack(images)


    # def process(self):
    #     data_list = []
    #     json_file_list = getJsonFile(self.root)
    #     for json_path in tqdm(json_file_list):
    #         dict = extract_teeth(json_path)
    #         label = []
    #         images_list = []
    #         pos_list = []
    #         for k in dict.keys():
    #             lbl = int(k)-1
    #             if lbl > 13 or lbl < 0:
    #                 continue
    #             label.append(lbl)
    #             pos_list.append([dict[k]['c_x'], dict[k]['c_y']])
    #             images_list.append(dict[k]['image'])
    #         images = self._get_features_from_images(images_list)
    #         labels = self._get_labels(label)
    #         pos = self._get_pos(pos_list)
    #         data = Data(x=images, y=labels, pos=pos)
    #         data_list.append(data)
    #     if self.pre_filter is not None:
    #         data_list = [d for d in data_list if self.pre_filter(d)]
    #
    #     if self.pre_transform is not None:
    #         data_list = [self.pre_transform(d) for d in data_list]
    #
    #     torch.save(self.collate(data_list), self.processed_paths[0])
    #
    #
    # def _get_features_from_images(self, list):
    #     images = []
    #     for i in list:
    #         image = cv2.resize(i, (256, 256))
    #         images.append(image)
    #     images = np.asarray(images)
    #     return torch.tensor(images, dtype=torch.float)

    def p_get_features_from_images(self, list):
        images = []
        for i in list:
            image = Image.open(i)
            image = image.resize((100, 100))
            image = np.array(image)
            image = image.ravel()
            images.append(image)
        images = np.asarray(images)

        return torch.tensor(images, dtype=torch.float)

    def _get_labels(self, label):
        label = np.asarray(label)
        return torch.tensor(label, dtype=torch.int64)

    def _get_pos(self, pos):
        pos = np.asarray(pos)
        return torch.tensor(pos, dtype=torch.int64)






