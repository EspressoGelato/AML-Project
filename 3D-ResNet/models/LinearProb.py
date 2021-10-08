import math
from collections import OrderedDict
import torch
import torch.nn as nn
from model import generate_model, load_pretrained_model
from collections import OrderedDict


class ModelPlusLinearLayers(nn.Module):
    """
    Linear Prob for learning downstream tasks
    In Channel: 512
    Out Channle: 512
    """

    def __init__(self, opt, in_channels, out_channels, layers=2):
        super(ModelPlusLinearLayers, self).__init__()
        self.pretrained_model = generate_model(opt, use_features=True)

        if self.training:
            self.pretrained_model = self.resume_model_from_pretraind(opt.video_pretrained_path)

        self.opt = opt
        if opt.linear_prob:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        elif opt.finetune_video:
            self.pretrained_model = self.pretrained_model

        layers = []
        midddle_dim = 512
        self.fc1 = nn.Linear(in_channels, midddle_dim)
        self.fc2 = nn.Linear(midddle_dim, out_channels)

        self.linear = nn.Linear(512, opt.n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x, f = self.pretrained_model(x)
        if self.opt.linear_prob:
            x = x.detach()
            f = f.detach()

        x = self.relu(self.fc1(f))
        #x = self.dropout(x)
        x = self.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.linear(x)

        return x


    def resume_model_from_pretraind(self, path):
        print("#"*100)
        print(path)
        state_dict = torch.load(path, map_location='cpu')
        model_state = state_dict['model']
        new_state_dict = OrderedDict()

        #model_dict_new = self.pretrained_model.state_dict()

        for k, v in model_state.items():
            if 'textual' in k:
                continue
            else:
                name = k[7:] 
                if name == 'linear.weight' or name == 'linear.bias':
                    continue
                new_state_dict[name] = v

        self.pretrained_model.load_state_dict(new_state_dict)

        return self.pretrained_model