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
from multiprocessing.sharedctypes import Value
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
import random


# Helper funcs
def cvt_for_plt(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def normalize(image_mat):
    return cv2.normalize(image_mat,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

def write_image_output(suffix,file_name,extension,image,path=None,display=False):
    ## TODO need to modify to write to current path
    output_file_path = ""
    if path is not None:
        output_file_path += path
        output_file_path += "output_"
    output_file_path += suffix + file_name + extension
    print("Writing "+suffix+" to file:",output_file_path)
    if display:
        cv2.imshow(suffix,image)
        cv2.waitKey(0)
    else:
        cv2.imwrite(output_file_path,image)

def calc_and_print_stats(suffix, image):
    arr = image.flatten()
    maxi = max(arr)
    mini = min(arr)
    avg = np.average(arr)
    stddev = np.std(arr)
    print(f"{suffix} mean:{avg}")
    print(f"{suffix} max:{maxi}")
    print(f"{suffix} min:{mini}")
    print(f"{suffix} dev:{stddev}")
    return [max,min,avg,stddev]

def standardize_image(image):
    mean = np.mean(image,keepdims=True)
    std = np.sqrt((image - mean)**2).mean(keepdims=True)
    return (image - mean)/std

def draw_markers(suffix,image,coords,color,display=False,file_name=None,output_file_extension=".png",path=None):
    image_dup = np.copy(image)
    for coord in coords:
        cv2.drawMarker(image_dup,[coord[1],coord[0]],color)
    if display:
        cv2.imshow(suffix,image_dup)
        cv2.waitKey(0)
    if not display:
        if file_name is None or path is None:
            raise ValueError("file_name or path is None when saving the image")
        write_image_output(suffix,file_name,output_file_extension,image_dup,path)

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

def apply_anms_to_img(local_maxima_coords,corner_score_img,n_best):
    "input: "
    "output: numpy array of coordinates"

    # initializing distances array
    distances = np.full(local_maxima_coords.shape[0],fill_value=CoOrds(0,0))
    for i,coords in enumerate(local_maxima_coords):
        distances[i] = CoOrds(coords[0],coords[1])

    ED = None
    for i,coord_i in enumerate(local_maxima_coords):
        for coord_j in local_maxima_coords:
            # print(i," ", corner_score_img[coord_j[0],coord_j[1]], " ", corner_score_img[coord_i[0],coord_i[1]])
            if (corner_score_img[coord_j[0],coord_j[1]] > corner_score_img[coord_i[0],coord_i[1]]):
                ED = (coord_j[0] - coord_i[0])**2 + (coord_j[1]-coord_i[1])**2
            if ED is not None and ED < distances[i].distance_score:
                distances[i].distance_score = ED
    
    sort_distances_obj = np.array(sorted(distances,reverse=True))
    print(f"Taking {n_best} out of {sort_distances_obj.shape}")
    sort_distances_obj = sort_distances_obj[0:n_best]
    
    anms_coords = []
    for coord in sort_distances_obj:
        anms_coords.append([coord.x,coord.y])

    anms_coords = np.array(anms_coords)
    return anms_coords

class FeatureDescriptor(cv2.KeyPoint):
    def __init__(self,x,y,corner_patch):
        super().__init__(int(y),int(x),15)
        self.feature_descriptor = corner_patch
    def __str__(self):
        return str(self.pt)
    def getPatch(self):
        return self.feature_descriptor

def get_corner_descriptors(img_gray: List[List], corner_locs: np.array(List[List])) -> List[FeatureDescriptor]:
    corner_descriptors = []
    for (x, y) in corner_locs:
        if x-20 <= 0 or x+20>img_gray.shape[0] or y-20<=0 or y+20>img_gray.shape[1]:
            continue
        corner_patch = img_gray[x-20:x+20, y-20:y+20]

        "subsampling every 25th pixel"
        corner_patch = corner_patch.flatten()[::25]
        corner_patch = np.reshape(corner_patch,newshape=(8,8))
        corner_patch = standardize_image(corner_patch)
        corner_patch = cv2.GaussianBlur(corner_patch, (3, 3), 1)
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

def get_feature_matches(img1_descriptor: List[FeatureDescriptor], img2_descriptor: List[FeatureDescriptor], match_thresh: float = 0.75) -> List[List]:
    feature_matches = []
    for point1 in img1_descriptor:
        min_dist = math.inf
        second_min_dist = math.inf
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
    Parser.add_argument('--BasePath',default='../Data/Train',help='Base path to get images from,Default:../Data/Train/')
    Parser.add_argument('--TestSet',default='Set1/',help='Test set to run algorithm on,Default:Set1')
    Parser.add_argument('--CornerHarrisBlockSize', default=4, help='Block Size for Harris Corner Detection, Default:7')
    Parser.add_argument('--CornerHarrisSobelOpSize', default=5, help='Block Size for Harris Corner Detection, Default:11')
    Parser.add_argument('--CornerHarrisKParameter', default=0.04, help='Block Size for Harris Corner Detection, Default:0.04')
    Parser.add_argument('--StdDevThresholdFactorForLocalMaxima',default=1,help='threshold for local maxima based on standard deviation of standardized corner score,Default:1')
    Parser.add_argument('--RansacMaxIterations',default=10000,help='Maximum number of iterations a RANSAC algorithm should run, Default:10000')

    Args = Parser.parse_args()
    ch_block_size = Args.CornerHarrisBlockSize
    ch_ksize = Args.CornerHarrisSobelOpSize
    ch_k = Args.CornerHarrisKParameter
    n_max_ransac_iterations = Args.RansacMaxIterations
    local_maxima_stddev_threshold_factor = Args.StdDevThresholdFactorForLocalMaxima
    base_path = Args.BasePath
    img_set = Args.TestSet
    input_file_extension = '.jpg'
    output_file_extension = '.png'
    
    """
    Read a set of images for Panorama stitching
    """
    files = glob.glob(base_path + img_set + "*" + input_file_extension,recursive=False)
    files = [file.replace("\\","/") for file in files]
    print("List of files to read:",files)
    file_names = [file.replace(base_path+img_set,'').replace(input_file_extension,'') for file in files]
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
    corners_of_images = []
    n_harris_corners_images = []
    min_stddev = None
    for i,file in enumerate(files):
        corners_of_img = cv2.cornerHarris(images_gray[i],ch_block_size,ch_ksize,ch_k) ### TODO need to try out Shi-Tomasi Corner Detection instead
        corners_of_img = standardize_image(corners_of_img)
        n_harris_corners = np.sum(corners_of_img>0.01*corners_of_img.max())
        [_,_,_,stddev] = calc_and_print_stats(file_names[i],corners_of_img)
        if min_stddev is not None and stddev < min_stddev:
            min_stddev = stddev
        else:
            min_stddev = stddev

        corners_of_images.append(corners_of_img)
        n_harris_corners_images.append(n_harris_corners)
    
        write_image_output("pure_corners",file_names[i],output_file_extension,normalize(corners_of_img),base_path+img_set)
    print(f"minimum standard deviation:{min_stddev}")
    
    local_max_corners_of_images = []
    n_local_max_corners_images = []
    local_max_corners_coords_of_images = []
    for i,file in enumerate(files):
        local_maxima_coords = peak_local_max(corners_of_images[i],min_distance=3,threshold_abs=min_stddev*local_maxima_stddev_threshold_factor)
        corner_local_maxima = np.zeros_like(corners_of_images[i])
        corner_local_maxima[tuple(local_maxima_coords.T)] = 1
        
        write_image_output("max_corners_locs",file_names[i],output_file_extension,normalize(corner_local_maxima),base_path+img_set)
        corner_local_maxima[tuple(local_maxima_coords.T)] = corners_of_images[i][tuple(local_maxima_coords.T)]
        write_image_output("max_corners",file_names[i],output_file_extension,normalize(corner_local_maxima),base_path+img_set)

        local_max_corners_of_images.append(corner_local_maxima)

        n_local_max_corners_images.append(local_maxima_coords.size)
        local_max_corners_coords_of_images.append([local_maxima_coords])

    local_max_corners_of_images = np.array(local_max_corners_of_images)
    n_local_max_corners_images = np.array(n_local_max_corners_images)
    local_max_corners_coords_of_images = np.array(local_max_corners_coords_of_images)
    for i,file in enumerate(files):
        draw_markers("embed_max_corners",images_color[i],local_max_corners_coords_of_images[i][0],color=[0,255,0],file_name=file_names[i],output_file_extension=output_file_extension,path=base_path+img_set,display=False)


    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    anms_coords_of_images = []
    for i,file in enumerate(files):
        anms_coords_of_images.append(apply_anms_to_img(local_max_corners_coords_of_images[i][0],local_max_corners_of_images[i],200))

    anms_coords_of_images = np.array(anms_coords_of_images)
    for i,file in enumerate(files):
        draw_markers("anms_corners",images_color[i],anms_coords_of_images[i],color=[0,255,0],file_name=file_names[i],output_file_extension=output_file_extension,path=base_path+img_set,display=False)

    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    images_corner_descriptors = []
    for i,file in enumerate(files):
        images_corner_descriptors.append([get_corner_descriptors(images_gray[i],anms_coords_of_images[i])])
    
    rand_img_i = random.randint(0,len(files)-1)
    rand_img_corner_j = random.randint(0,len(images_corner_descriptors[rand_img_i][0])-1)
    rand_patch = images_corner_descriptors[rand_img_i][0][rand_img_corner_j].getPatch()
    rand_patch = np.reshape(rand_patch,newshape=(8,8))
    write_image_output(f"image{rand_img_i}_corner{rand_img_corner_j}",file_names[rand_img_i],
            output_file_extension,
            rand_patch,
            base_path+img_set,display=False)

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    images_feature_matches = []
    for first_image_idx, file in enumerate(files):
        for second_image_idx in range(first_image_idx+1,len(files)):
            feature_matches = get_feature_matches(images_corner_descriptors[first_image_idx][0],images_corner_descriptors[second_image_idx][0])
            images_feature_matches.append(feature_matches)
            
            """
            Below portion of the code is to draw image correlations
            """
            keypoints1 = []
            keypoints2 = []
            matches = []
            for i,feature_descriptor_dmatch in enumerate(feature_matches):
                keypoints1.append(feature_descriptor_dmatch.keypoint1)
                keypoints2.append(feature_descriptor_dmatch.keypoint2)
                matches.append(cv2.DMatch(i,i,feature_descriptor_dmatch.keypoint_distance))
            keypoints1 = np.array(keypoints1)
            keypoints2 = np.array(keypoints2)

            ret = np.array([])
            drew_image = cv2.drawMatches(img1=images_color[first_image_idx],
                keypoints1=keypoints1,
                img2=images_color[second_image_idx],
                keypoints2=keypoints2,
                matches1to2=matches,outImg = ret,matchesThickness=1)
            write_image_output(f"feature_matches_",f"{first_image_idx}_{second_image_idx}",output_file_extension,drew_image,base_path+img_set)

    

    
    """
    Refine: RANSAC, Estimate Homography
    """
    homography_mats = []
    for i,image_pair_feature_matches in enumerate(images_feature_matches):
        current_max_inliers = []
        for i in range(n_max_ransac_iterations):
            if len(current_max_inliers) > len(images_feature_matches):
                break
            random



    
    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """


if __name__ == "__main__":
    main()
