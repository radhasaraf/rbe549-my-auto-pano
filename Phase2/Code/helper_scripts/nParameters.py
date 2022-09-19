import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import sys
sys.path.insert(1,'../Network/')
from Network import *
from torchvision.transforms import ToTensor
from Network import CIFAR10Model
from torchsummary import summary

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = CIFAR10Model(CIFAR10Model.Model.DenseNet) 
model = model.to(device)


base_path = "../../../../cnn_training/densenet/train_29_08_2022_4/"
ModelPath = base_path + "/Checkpoints/19model.ckpt"

CheckPoint = torch.load(ModelPath)
model.load_state_dict(CheckPoint['model_state_dict'])
print('Number of parameters in this model are %d ' % len(model.state_dict().items()))
# print('Number of parameters in this model are %d ' % len(model.parameters()))

nparameter = 0
for parameter in model.parameters():
    nparameter += 1

summary(model,(3,32,32))
