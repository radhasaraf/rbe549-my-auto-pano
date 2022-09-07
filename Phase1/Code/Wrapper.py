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
import glob
import sys
import argparse
from matplotlib import pyplot as plt
from skimage.feature import corner_peaks,peak_local_max
from typing import List
import math


# Helper funcs
def cvt_for_plt(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def make_subplots(row_length,images):
    print(len(images))
    if len(images)%5 == len(images):
        fig,axs = plt.subplots(len(images))
        fig.set_size_inches(16,16)
        axs = axs.ravel()    
    else:
        fig,axs = plt.subplots(5,int(len(images)/5))
        fig.set_size_inches(16,16)
        axs = axs.ravel()    

    for i,image in enumerate(images):
            axs[i].imshow(cvt_for_plt(image))

def embed_corners(img,corner_score_img=None,points=None):
    if corner_score_img is None and points is None:
        return
    img_corners = np.copy(img)
    if corner_score_img is not None:
        corner_score_img1 = cv2.dilate(corner_score_img,None)
        img_corners[corner_score_img1>0.01*corner_score_img1.max()] = [0,0,255]
        print(np.sum(corner_score_img1>0.01*corner_score_img1.max()))
    elif points is not None:
        print(np.shape(points))
        img_corners[tuple(points.T)] = [0,0,255]
    return img_corners

def embed_and_plot_corners(color_images,images_corner_score=None,images_corner_points=None):
    if images_corner_score is None and images_corner_points is None:
        return
    corner_embed_images = []
    if images_corner_score is not None:
        for i in range(len(color_images)):
            corner_embed_images.append(embed_corners(color_images[i],corner_score_img=images_corner_score[i]))
    elif images_corner_points is not None:
        for i in range(len(color_images)):
            corner_embed_images.append(embed_corners(color_images[i],points=images_corner_points[i]))
    make_subplots(5,corner_embed_images)

## ANMS
from functools import total_ordering


@total_ordering
class CoOrds:
    def __init__(self,x,y):
        self.x = int(x)
        self.y = int(y)
        self.distance_score = sys.float_info.max
    def __lt__(self,obj):
        return ((self.distance_score) < (obj.distance_score))
    def __repr__(self) -> str:
        return "(x,y,distance):("+str(self.x)+","+str(self.y)+","+str(self.distance_score)+")"

def apply_anms_to_img(corner_score_img,n_best,anms_local_maxima_threshold):
    "input: "
    "output: numpy array of coordinates"
    local_maxima_coords = peak_local_max(corner_score_img,min_distance=3,threshold_abs=anms_local_maxima_threshold)

    # initializing distances array
    distances = np.full(local_maxima_coords.shape[0],fill_value=CoOrds(0,0))
    for i,coords in enumerate(local_maxima_coords):
        distances[i] = CoOrds(coords[0],coords[1])


    ED = None
    for i,coord_i in enumerate(local_maxima_coords):
        for coord_j in local_maxima_coords:
            # print(i," ", corner_score_img[coord_j[0],coord_j[1]], " ", corner_score_img[coord_i[0],coord_i[1]])
            if (corner_score_img[coord_j[0],coord_j[1]] > corner_score_img[coord_i[0],coord_i[1]]):
                ED = np.power((coord_j[0] - coord_i[0]),2) + np.power((coord_j[1]-coord_i[1]),2)
            if ED is not None and ED < distances[i].distance_score:
                distances[i].distance_score = ED
    sort_distances_obj = np.array(sorted(distances,reverse=True))
    print(sort_distances_obj.shape)
    sort_distances_obj = sort_distances_obj[0:n_best]
    
    
    sorted_dists_mat = np.zeros(shape=(sort_distances_obj.shape[0],2)) 

    for i,dist in enumerate(sort_distances_obj):
        sorted_dists_mat[i] = np.array([int(sort_distances_obj[i].x), sort_distances_obj[i].y])
    return sorted_dists_mat.astype(int)

class FeatureDescriptor(cv2.KeyPoint):
    def __init__(self,x,y,corner_patch):
        super().__init__(int(x),int(y),15)
        self.feature_descriptor = corner_patch
    def __str__(self):
        return str(self.pt)

def get_corner_descriptors(img_gray: List[List], corner_locs: np.array(List[List])) -> List[FeatureDescriptor]:
    corner_descriptors = []
    for (x, y) in corner_locs:
        if x-20 < 0 or x+20>img_gray.shape[0] or y-20<0 or y+20>img_gray.shape[1]:
            continue
        corner_patch = img_gray[x-20:x+20, y-20:y+20]
        corner_patch = cv2.GaussianBlur(corner_patch, (3, 3), 0)
        corner_patch = cv2.resize(corner_patch, (8, 8), interpolation=cv2.INTER_AREA)
        feature_descriptor = FeatureDescriptor(x,y,corner_patch.flatten())
        corner_descriptors.append(feature_descriptor)
    return corner_descriptors


class DMatchWrapper():
    def __init__(self,point1,point2,distance) -> None:
        self.keypoint1 = point1
        self.keypoint2 = point2
        self.keypoint_distance = distance
    def __str__(self) -> str:
        return str(self.keypoint1)+str(self.keypoint2)
def get_feature_matches(img1_descriptor: List[FeatureDescriptor], img2_descriptor: List[FeatureDescriptor], match_thresh: float = 0.9) -> List[List]:
    feature_matches = []
    for point1 in img1_descriptor:
        min_dist = second_min_dist = math.inf
        match_point2 = None
        for point2 in img2_descriptor:
            dist = math.dist(point1.feature_descriptor, point2.feature_descriptor)
            if dist < min_dist:
                second_min_dist = min_dist
                min_dist = dist
                match_point2 = point2
        if (min_dist / second_min_dist) < match_thresh:
            feature_matches.append(DMatchWrapper(point1, match_point2,min_dist))

    return feature_matches

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CorerHarrisBlockSize', default=4, help='Block Size for Harris Corner Detection, Default:7')
    Parser.add_argument('--CornerHarrisSobelOpSize', default=5, help='Block Size for Harris Corner Detection, Default:11')
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
    
    files = []
    for file in glob.glob(base_path + img_set + "*.jpg",recursive=False):
        files.append(file.replace("\\","/"))
    print(files)
    # images_color = np.array(shape=(1len(files))
    images_color = []
    images_gray = []
    for image_file in files:
        img_color_orig = cv2.imread(image_file)
        img_gray = cv2.cvtColor(img_color_orig,cv2.COLOR_BGR2GRAY)
        images_color.append(img_color_orig)
        images_gray.append(img_gray)

    images_color = np.array(images_color)
    print(images_color.shape)
    # make_subplots(len(files),images_color)

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    cornersOfImages = []
    n_harris_corners_images = []
    for i,image in enumerate(images_gray):
        cornersOfImg = cv2.cornerHarris(image,ch_block_size,ch_ksize,ch_k) ### TODO need to try out Shi-Tomasi Corner Detection instead
        n_harris_corners = np.sum(cornersOfImg>0.01*cornersOfImg.max())
        
        cornersOfImages.append(cornersOfImg)
        n_harris_corners_images.append(n_harris_corners)

    cornersOfImages = np.array(cornersOfImages)
    n_harris_corners_images = np.array(n_harris_corners_images)
    # embed_and_plot_corners(images_color,cornersOfImages)


    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    anms_distances = []
    for i in range(len(images_color)):
        anms_distances.append(apply_anms_to_img(cornersOfImages[i],1000,anms_local_maxima_threshold))

    anms_distances = np.array(anms_distances)
    # embed_and_plot_corners(images_color,images_corner_points=anms_distances)

    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    images_corner_descriptors = []
    for i in range(len(images_color)):
        images_corner_descriptors.append(get_corner_descriptors(images_gray[i],anms_distances[i]))

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    feature_matches = get_feature_matches(images_corner_descriptors[0],images_corner_descriptors[1])
    feature_points = []
    keypoints1 = []
    keypoints2 = []
    matches = []
    for i,feature_descriptor_dmatch in enumerate(feature_matches):
        keypoints1.append(feature_descriptor_dmatch.keypoint1)
        keypoints2.append(feature_descriptor_dmatch.keypoint2)
        matches.append(cv2.DMatch(i,i,feature_descriptor_dmatch.keypoint_distance))
    keypoints1 = np.array(keypoints1)
    keypoints2 = np.array(keypoints2)
    print(keypoints1[0].pt)
    print(keypoints2[0].pt)

    ret = np.array([])
    drew_image = cv2.drawMatches(img1=images_color[0],
        keypoints1=keypoints1,
        img2=images_color[1],
        keypoints2=keypoints2,
        matches1to2=matches,outImg = ret,matchesThickness=1)

    """
    Refine: RANSAC, Estimate Homography
    """

    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """


if __name__ == "__main__":
    main()
