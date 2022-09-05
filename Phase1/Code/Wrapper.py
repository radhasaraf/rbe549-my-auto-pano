#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2

# Add any python libraries here
import argparse
import cv2

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CorerHarrisBlockSize', default=4, help='Block Size for Harris Corner Detection, Default:4')
    Parser.add_argument('--CornerHarrisSobelOpSize', default=5, help='Block Size for Harris Corner Detection, Default:5')
    Parser.add_argument('--CornerHarrisKParameter', default=0.04, help='Block Size for Harris Corner Detection, Default:0.04')
    Parser.add_argument('--ANMSLocalMaximaThreshold', default=0.01, help='ANMS Local maxima threshold, Default:0.01')

    Args = Parser.parse_args()
    ch_block_size = Args.CornerHarrisBlockSize
    ch_ksize = Args.CornerHarrisSobelOpSize
    ch_k = Args.CornerHarrisKParameter
    ansm_local_maxima_threshold = Args.ANMSLocalMaximaThreshold

    """
    Read a set of images for Panorama stitching
    """
    base_path = '../Data/Train/'
    img_set = 'Set1/'
    img_sequence = '3'
    img1 = cv2.imread(base_path + img_set + img_sequence + '.jpg')
    

    """
    Corner Detection
    Save Corner detection output as corners.png
    """

    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """

    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """

    """
    Refine: RANSAC, Estimate Homography
    """

    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """


if __name__ == "__main__":
    main()
