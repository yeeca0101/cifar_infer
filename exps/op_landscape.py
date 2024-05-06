# copy by https://github.com/digantamisra98/Mish/blob/master/exps/op_landscape.py

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torch import nn

# my external location
import sys
sys.path.append('../../')
from experiments.activation.acts import *

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)

class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()

    def forward(self, input):
        return input * torch.tanh(F.softplus(input))
    
class ActivationModule(nn.Module):
    def __init__(self, activation_func):
        super().__init__()
        self.activation_func = activation_func
    
    def forward(self, x):
        return self.activation_func(x)

def build_model(input_size, output_sizes, activation_func,normalize=True):
    layers = [("fc1", nn.Linear(input_size, output_sizes[0]))]
    if normalize:
        layers.append((f"norm1", nn.LayerNorm(output_sizes[0])))

    for i, size in enumerate(output_sizes):
        if i > 0:
            layers.append((f"fc{i+1}", nn.Linear(output_sizes[i-1], size)))
            if normalize:
                layers.append((f"norm{i+1}", nn.LayerNorm(size)))
        layers.append((f"activation{i+1}", ActivationModule(activation_func)))
    
    return nn.Sequential(OrderedDict(layers))

def generate_activation_landscape(model, resolution, range_x, range_y,gpu=True):
    x = np.linspace(range_x[0], range_x[1], num=resolution)
    y = np.linspace(range_y[0], range_y[1], num=resolution)
    device = torch.device('cuda:0' if gpu else 'cpu')
    to_device = lambda x : x.to(device)
    grid = [to_device(torch.tensor([xi, yi], dtype=torch.float32)) for xi in x for yi in y]
    model.to(device)
    with torch.no_grad():
        outputs = torch.stack([model(point) for point in grid]).cpu()
    return outputs.detach().numpy().reshape(resolution, resolution)

def convert_to_PIL(img):
    scaler = MinMaxScaler(feature_range=(0, 255))
    img = scaler.fit_transform(img)
    return Image.fromarray(img)

def main_with_logging(activations, resolution=400, range_x=(-10.0, 40.0), range_y=(-10.0, 40.0)):
    wandb.init(project="SwishT_Landscapes") 
    wandb_dict = {
        "Output landscapes": []
    }

    models = {name: build_model(2, [64, 32, 16, 1], func) for name, func in activations.items()}
    landscapes = {}
    for name, model in models.items():
        np_img = generate_activation_landscape(model, resolution, range_x, range_y)
        pil_img = convert_to_PIL(np_img)
        landscapes[name] = pil_img
        # Log images to wandb
        wandb_dict["Output landscapes"].append(wandb.Image(np_img, caption=name))
    
    wandb.log(wandb_dict)

    return landscapes

if __name__ == "__main__":

    activations = {
        "ReLU": F.relu,
        "Swish": lambda x: x * torch.sigmoid(x),
        "Mish": lambda x: x * torch.tanh(F.softplus(x)),
        'SwishT': SwishT(beta_init=1.),
        'SwishT_trained': SwishT(beta_init=6.4),
        'GELU':F.gelu
    }

    main_with_logging(activations,)
