import os
import torch
import torch.nn as nn
from collections import OrderedDict


def get_model_path(model_name):
    model_file=model_name+'.pth'
    curr_dict = os.getcwd()
    print(curr_dict)
    cache_dir = os.path.join(curr_dict, 'models')
    os.makedirs(cache_dir, exist_ok=True)
    fpath=os.path.join(cache_dir,model_file)
    if not os.path.isfile(fpath):
        print('Model not exist.')
    return fpath     

class MTNet(nn.Module):
    def __init__(self, num_emotion, num_race = 6, num_sex=2):
        super().__init__()
        curr_dict = os.getcwd()
        cache_dir = os.path.join(curr_dict, 'HRI/models')
        fpath = '_HAR/HRI/models/enet_b2_7.pt'

        self.base_net = torch.load(fpath)
        self.in_dim = self.base_net.classifier.in_features
        self.base_net.classifier = nn.Identity()
        
        self.classifier1 = nn.Sequential(OrderedDict([('linear1', nn.Linear(self.in_dim,self.in_dim)),('Prelu1', nn.PReLU()),('final', nn.Linear(self.in_dim, num_emotion))]))
        self.classifier2 = nn.Sequential(OrderedDict([('linear1', nn.Linear(self.in_dim,self.in_dim)),('Prelu1', nn.PReLU()),('final', nn.Linear(self.in_dim, num_race))]))
        self.classifier3 = nn.Sequential(OrderedDict([('linear1', nn.Linear(self.in_dim,self.in_dim)),('Prelu1', nn.PReLU()),('final', nn.Linear(self.in_dim, num_sex))]))
        
    def forward(self, x):
        features = self.base_net(x)
        emo_head = self.classifier1(features)
        race_head = self.classifier2(features)
        sex_head = self.classifier3(features)

        return emo_head, race_head, sex_head
        

    
