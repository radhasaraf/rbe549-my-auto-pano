import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import sys
sys.path.insert(1,'../Network/')
from Network import *
from torchvision.transforms import ToTensor

from Network import CIFAR10Model

# Writer will output to ./runs/ directory by default
writer = SummaryWriter('../../../../cnn_training/densenet/train_29_08_2022_4/graph/')

normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])
# transforms_to_apply = transforms.Compose([transforms.Resize((64,64)),
#                                         transforms.ToTensor(),
#                                         normalize])

transforms_to_apply = transforms.Compose([transforms.ToTensor(),
                                        normalize])                                        

TrainSet = torchvision.datasets.CIFAR10(root='../../../data', train=True,download=False, transform=transforms_to_apply)

trainloader = torch.utils.data.DataLoader(TrainSet, batch_size=16, shuffle=True)
model = CIFAR10Model(CIFAR10Model.Model.DenseNet)
model.eval()
images, labels = next(iter(trainloader))

grid = torchvision.utils.make_grid(images)
# writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()