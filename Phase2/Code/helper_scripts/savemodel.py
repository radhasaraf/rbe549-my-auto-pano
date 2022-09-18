#  For existing network models , But no training has been saved  .pth  File status 
import netron
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import sys
sys.path.insert(1,'../Network/')
# from Network import *
from torchvision.transforms import ToTensor
from Network_supervised import Net
from torchsummary import summary

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = Net() 
torch.save(model,'homography_supervised.pt')
# model = model.to(device)

# x = torch.randn(16, 3, 64, 64).to(device)  #  Randomly generate an input 
# modelData = "./resnet.pth"  #  Define the path where the model data is saved 
# # modelData = "./demo.onnx" #  Some people say it should be  onnx  file , But I tried  pth  Yes.  
# torch.onnx.export(model, x, modelData)  #  take  pytorch  Model with  onnx  Export in format and save 
# netron.start(modelData)  #  Output network structure 

# #  For existing network models  .pth  File status 
# import netron

# modelData = "./demo.pth"  #  Define the path where the model data is saved 
# netron.start(modelData)  #  Output network structure 