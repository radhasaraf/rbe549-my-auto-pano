import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import sys
sys.path.insert(1,'../Network/')
from Network_supervised import HomographyModel
from torchvision.transforms import ToTensor
from Train import GenerateBatch
import numpy as np
import cv2
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")


def get_homo(BasePath,TrainCoordinates, MiniBatchSize=1):

    # Choose a random image
    random_orig_name = "1"
    orig_img_path = os.path.join(BasePath, f"{set}\\Orig", random_orig_name)
    warped_img_path = os.path.join(BasePath, f"{set}\\Warped", random_orig_name)

    img1 = cv2.imread(orig_img_path)
    img2 = cv2.imread(warped_img_path)

    stacked_images = np.float32(np.concatenate([img1, img2], axis=2))

    # Get label
    h4pt = TrainCoordinates[random_orig_name]
    print(f"h4pt:{h4pt}")

    # Append All Images and Mask
    stacked_images_batch = []
    h4pt_batch = []
    stacked_images_batch.append(torch.from_numpy(stacked_images))
    h4pt_batch.append(torch.tensor(h4pt))

    model = HomographyModel().to(device)
    model.eval()
    PredicatedCoordinatesBatch = model(stacked_images_batch)
    print(f"PredicatedCoordinatesBatch:{PredicatedCoordinatesBatch}")

    patch_size = 128
    patch_corners = np.array([(0, 0), (patch_size, 0), (0, patch_size), (patch_size, patch_size)])
    perturbations = PredicatedCoordinatesBatch.reshape(1,4,2)[0]
    perturbed_corners = patch_corners + perturbations



    H,_ = cv2.findHomography(patch_corners,perturbed_corners)
    print(H)



