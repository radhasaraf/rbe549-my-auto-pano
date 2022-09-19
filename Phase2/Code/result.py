import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import sys
sys.path.insert(1,'../Network/')
from Network.Network_supervised import HomographyModel
from torchvision.transforms import ToTensor
from Train import GenerateBatch
import numpy as np
import cv2
import os
import csv
import ast

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")


def get_homo(BasePath,TrainCoordinates, set="Raw", MiniBatchSize=1):

    # Choose a random image
    random_orig_name = "1039_1.jpg"
    orig_img_path = os.path.join(BasePath, f"{set}\\Orig", random_orig_name)
    warped_img_path = os.path.join(BasePath, f"{set}\\Warped", random_orig_name)
    print(orig_img_path)
    print(warped_img_path)

    img =  cv2.imread("..\\Data\\Train\\1039.jpg")
    h,w = img.shape[0],img.shape[1]
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
    # stacked_images_batch = torch.stack(stacked_images_batch).to(device)
    # h4pt_batch.append(torch.tensor(h4pt))

    # model = HomographyModel().to(device)
    # model.eval()
    # PredicatedCoordinatesBatch = model(stacked_images_batch)
    # print(f"PredicatedCoordinatesBatch:{PredicatedCoordinatesBatch}")

    # PredicatedCoordinatesBatch = PredicatedCoordinatesBatch.to("cpu")
    # perturbations = PredicatedCoordinatesBatch.reshape(1,4,2)[0].detach().numpy()
    patch_size = 128
    patch_corners = np.array([(0, 0), (patch_size, 0), (0, patch_size), (patch_size, patch_size)])
    perturbations = np.array([-12, -9, -19, 14, 12, -12, 15, 30]).reshape(4,2)
    perturbed_corners = patch_corners + perturbations

    H,_ = cv2.findHomography(patch_corners,perturbed_corners)
    transformed_img = cv2.warpPerspective(img, np.linalg.inv(H), (h, w), flags=cv2.INTER_LINEAR)
    cv2.imwrite("original.png",img)
    cv2.imwrite("model_warped.png",transformed_img)

    orig_pertubations = np.array([-23, -18, -29, 29, -5, -22, 5, 32]).reshape(4,2)
    original_warped = patch_corners + orig_pertubations
    H_orig,_ = cv2.findHomography(patch_corners,original_warped)
    original_warped_img = cv2.warpPerspective(img, np.linalg.inv(H_orig), (h, w), flags=cv2.INTER_LINEAR)

    cv2.imwrite("original_warp.png",original_warped_img)

    print(H)

LabelsPathTrain = os.path.join('..\\Data\\labels.csv')
print(LabelsPathTrain)
with open(LabelsPathTrain, 'r') as labels_file:
    reader = csv.reader(labels_file)
    TrainCoordinates = {row[0]: ast.literal_eval(row[1]) for row in reader}

# print(f"coords:{TrainCoordinates}")
get_homo("..\\Data",TrainCoordinates)


