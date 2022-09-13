"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import os
import csv
import ast
import cv2
import numpy as np
import random
import skimage
import PIL
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll(BasePath, CheckPointPath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize
    Trainabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    # Setup DirNames
    DirNamesTrain = SetupDirNames(os.path.join(BasePath, "Train/Orig/"))

    # Read and Setup Labels
    LabelsPathTrain = os.path.join('../Data/labels.csv')
    TrainLabels = ReadLabels(LabelsPathTrain)

    # If CheckPointPath doesn't exist make the path
    if not (os.path.isdir(CheckPointPath)):
        os.makedirs(CheckPointPath)

    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 100
    # Number of passes of Val data with MiniBatchSize
    NumTestRunsPerEpoch = 5

    # Image Input Shape
    ImageSize = [32, 32, 3]
    NumTrainSamples = len(DirNamesTrain)

    # Number of classes
    NumClasses = 10

    return (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainLabels,
        NumClasses,
    )


def ReadLabels(path_to_csv):
    if not (os.path.isfile(path_to_csv)):
        print("ERROR: Train Labels do not exist in " + path_to_csv)
        sys.exit()
    else:
        with open(path_to_csv, 'r') as labels_file:
            reader = csv.reader(labels_file)
            img_homography_labels = {row[0]: ast.literal_eval(row[1]) for row in reader}
    return img_homography_labels


def SetupDirNames(BasePath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    # DirNamesTrain = ReadDirNames("./TxtFiles/DirNamesTrain.txt")

    return os.listdir(BasePath)


def ReadDirNames(ReadPath):
    """
    Inputs:
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, "r")
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames
