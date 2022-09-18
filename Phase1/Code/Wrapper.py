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
from gettext import translation
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
import warnings
from shapely.geometry import Point,Polygon


# Helper funcs
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

def calc_and_print_stats(image):
    arr = image.flatten()
    maxi = max(arr)
    mini = min(arr)
    try:
        avg = np.average(arr)
        stddev = np.std(arr,dtype=np.float32)
    except TypeError:
        print(f"error arr:{image.dtype}") 
        return [1,0,0,0]
    return [max,min,avg,stddev]

def standardize_image(image,debug=False):
    if debug:
        print(type(image))
    try:
        mean = np.mean(image,dtype=np.float32)
    except TypeError:
        return False,image
    std = np.sqrt((image - mean)**2).mean()
    return True,(image - mean)/std

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


## ANMS
from functools import total_ordering

def get_corners(img,bsize,ksize,k,file_name,output_extn,output_path):
    corners_of_img = cv2.cornerHarris(img,bsize,ksize,k) ### TODO need to try out Shi-Tomasi Corner Detection instead
    _,corners_of_img = standardize_image(corners_of_img)
    n_harris_corners = np.sum(corners_of_img>0.01*corners_of_img.max())
    [_,_,_,stddev] = calc_and_print_stats(corners_of_img)
    
    write_image_output("pure_corners",
            file_name,
            output_extn,
            normalize(corners_of_img),
            output_path)
    
    return corners_of_img,n_harris_corners,stddev

def apply_corners_local_maxima(corner_image,stddev,factor,file_name,output_ext,output_path):
    local_maxima_coords = peak_local_max(corner_image,min_distance=3,threshold_abs=stddev*factor)
    corner_local_maxima = np.zeros_like(corner_image)
    corner_local_maxima[tuple(local_maxima_coords.T)] = 1
    
    write_image_output("max_corners_locs",
            file_name,
            output_ext,
            normalize(corner_local_maxima),
            output_path)
    
    corner_local_maxima[tuple(local_maxima_coords.T)] = corner_image[tuple(local_maxima_coords.T)]
    
    write_image_output("max_corners",
            file_name,
            output_ext,
            normalize(corner_local_maxima),
            output_path)

    return corner_image,local_maxima_coords

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
    if len(local_maxima_coords) <=0:
        return local_maxima_coords

    assert(len(local_maxima_coords[0]) == 2)

    # initializing distances array
    distances = np.full(local_maxima_coords.shape[0],fill_value=CoOrds(0,0))
    for i,coords in enumerate(local_maxima_coords):
#        if len(coords) < 2:
#            print(coords)
#            continue
        x,y = coords
        ret = CoOrds(x,y)
        distances[i] = ret

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
        # print(corner_patch.shape)
        is_good, corner_patch = standardize_image(corner_patch,False)
        if not is_good:
            continue
        # corner_patch = cv2.GaussianBlur(corner_patch, (3, 3), 1)
        feature_descriptor = FeatureDescriptor(x,y,corner_patch.flatten())
        corner_descriptors.append(feature_descriptor)
    return True, corner_descriptors

def draw_random_corner_patch(img_fds,file_names,output_extension,output_path,display=False):
    rand_img_i = random.randint(0,len(file_names)-1)
    rand_img_corner_j = random.randint(0,len(img_fds[rand_img_i][0])-1)
    rand_patch = img_fds[rand_img_i][0][rand_img_corner_j].getPatch()
    rand_patch = np.reshape(rand_patch,newshape=(8,8))
    write_image_output(f"image{rand_img_i}_corner{rand_img_corner_j}",file_names[rand_img_i],
            output_extension,
            rand_patch,
            output_path,display=display)

def get_feature_matches(img1_descriptor: List[FeatureDescriptor], img2_descriptor: List[FeatureDescriptor], match_thresh: float = 0.75) -> List[List]:
    keypoints1 = []
    keypoints2 = []
    keypoint_distances = [] 
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
            keypoints1.append(point1)
            keypoints2.append(match_point2)
            keypoint_distances.append(min_dist)
    
    keypoints1 = np.array(keypoints1)
    keypoints2 = np.array(keypoints2)
    keypoint_distances = np.array(keypoint_distances) 
    return [keypoints1,keypoints2,keypoint_distances]

def draw_feature_matches(keypoints1,keypoints2,keypoints_distances,first_image,second_image,ref_file_id,to_file_id,output_file_extension,output_path,suffix=""):
    matches = []
    for i,feature_descriptor_dmatch in enumerate(keypoints_distances):
        matches.append(cv2.DMatch(i,i,keypoints_distances[i]))

    ret = np.array([])
    drew_image = cv2.drawMatches(img1=first_image,
        keypoints1=keypoints1,
        img2=second_image,
        keypoints2=keypoints2,
        matches1to2=matches,outImg = ret,matchesThickness=1)
    write_image_output(f"{suffix}feature_matches_",f"{ref_file_id}_{to_file_id}",output_file_extension,drew_image,output_path)

def perform_ransac(max_ransac_iters,first_kps,second_kps):
    kps_src = np.array([point.pt for point in first_kps])
    kps_dst = np.array([point.pt for point in second_kps])
    
    X = np.append(kps_src,np.ones(shape=(1,kps_src.shape[0])).T,axis=1)    

    max_inliers = []
    for iter in range(max_ransac_iters):
        if len(max_inliers) > len(kps_src): ## TODO percentage parameter for inliers_count
            break
        
        homography_points_inds = random.sample(range(len(kps_src)-1),4)
        homography_points_src = kps_src[homography_points_inds]
        homography_points_dst = kps_dst[homography_points_inds]
        
        H = cv2.findHomography(homography_points_src,homography_points_dst)  # TODO: Implement your own
        if (H[0] is None):
            continue
        X_proj = np.dot(H[0],(X.T))
        try:
            X_proj = np.array([X_proj[0]/X_proj[2],X_proj[1]/X_proj[2]]).T
        except RuntimeWarning:
            """
            projection division can give incorrect values
            """
            continue

        error = np.sum((kps_dst - X_proj)**2,axis=1)

        curr_iter_inliers = np.sum(error < 5) ## TODO threshold as parameter
        if curr_iter_inliers > np.sum(max_inliers):
            max_inliers = (error < 5)
    
    return max_inliers

def find_homography(ref_kps,to_kps,kps_inds):
    kps_src = np.array([point.pt for point in to_kps])
    kps_dst = np.array([point.pt for point in ref_kps])
    inlier_src = kps_src[kps_inds]
    inlier_dst = kps_dst[kps_inds]
    H,_ = cv2.findHomography(inlier_src,inlier_dst)
    return H

def get_img_coords(img_shape):
    base_coords = np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=list).T
    coords = base_coords.copy()
    X = coords[0]
    Y = coords[1]
    X[:] = X*img_shape[1] # this is because row and column are y and x
    Y[:] = Y*img_shape[0]
    return coords

def get_translated_img_bb_coords(coords,translation_vector,old_shape):
    translation_mat = np.identity(3)
    translation_mat[0:3,2] = translation_vector.T[0]
    new_img_bb_coords = np.dot(translation_mat,coords)
    new_image_shape = np.ceil(np.max(new_img_bb_coords, axis=1)).astype('int')[0:2]
    
    new_image_shape[0],new_image_shape[1] = new_image_shape[1],new_image_shape[0]

    print(f"After translation: image shape: \n {old_shape} --> {new_image_shape} \n bounding box coordinates: \n {new_img_bb_coords}")
    return new_img_bb_coords,new_image_shape

def get_tf_coords(H,img_shape):
    """
    get output image dimensions for full image after aplying Homography 
    this is needed for warpPerspective function
    """
    coords = get_img_coords(img_shape)
    coords_tf_bb = np.dot(H,coords)
    Z_new = coords_tf_bb[2]
    try:
        coords_tf_bb = coords_tf_bb/Z_new
    except RuntimeWarning:
        print("something is wrong")
        return None
    
    print(f"image bounding box coordinates changed from \n {coords[0:2]} \n to: \n {coords_tf_bb[0:2]}")
    translation_vec = -np.floor(
                        np.min(coords_tf_bb,
                            axis=1,
                            initial=0,    
                            where=coords_tf_bb<0,
                            keepdims=True)).astype('int')

    print(f"minimum translation needed {translation_vec.T}")

    translation_mat = np.identity(3)
    translation_mat[0:3,2] = translation_vec.T[0]

    new_image_bb_coords,new_image_shape = get_translated_img_bb_coords(coords_tf_bb,translation_vec,img_shape)
    
    new_image_coords = get_img_coords(new_image_shape)

    print(f"image shape coordinates changed from \n {coords[0:2]} \n to \n {new_image_coords[0:2]}")
    
    return [translation_vec,new_image_shape,new_image_bb_coords]

def get_full_homography_img(img,H):
    img_shape = img.shape
    [translation_vec,new_image_shape,new_image_bb_coords] = get_tf_coords(H,img_shape)
    translation_mat = np.zeros_like(H)
    translation_mat[0:3,2] = translation_vec.T[0]
    tf_homograph_mat = H + translation_mat

    tf_img = cv2.warpPerspective(img,tf_homograph_mat,[new_image_shape[1],new_image_shape[0]]) 
    return tf_img,translation_vec,new_image_shape,new_image_bb_coords

def affine_and_resize_image(image,translation_vec,new_shape):
    affine_mat = np.zeros(shape=(2,3))
    affine_mat[0,0] = 1
    affine_mat[1,1] = 1
    affine_mat[0:2,2] = translation_vec.T[0,0:2]

    coords = get_img_coords(image.shape)

    new_bb_coords,tf_shape = get_translated_img_bb_coords(coords,translation_vec,image.shape)

    final_canvas_shape = np.array([max(new_shape[0],tf_shape[0]),max(new_shape[1],tf_shape[1])])
    print(f"final canvase shape {new_shape},{tf_shape} --> {final_canvas_shape}")

    
    tf_ref_img = cv2.warpAffine(image,affine_mat,[final_canvas_shape[1],final_canvas_shape[0]])
    return tf_ref_img,new_bb_coords,final_canvas_shape

def get_image_info(ref_image_orig,Args,ref_file_id):
    print(f"\n\n Running get_image_info for {ref_file_id}")
    ref_image = ref_image_orig.copy()
    ch_block_size = Args.CornerHarrisBlockSize
    ch_ksize = Args.CornerHarrisSobelOpSize
    ch_k = Args.CornerHarrisKParameter
    n_max_ransac_iterations = Args.RansacMaxIterations
    local_maxima_stddev_threshold_factor = Args.StdDevThresholdFactorForLocalMaxima
    max_features =  Args.ANMSMaxFeatures
    base_path = Args.BasePath
    img_set = Args.TestSet
    input_file_extension = '.jpg'
    output_file_extension = '.png'
    out_path = base_path + img_set

    ref_img_gray = cv2.cvtColor(ref_image,cv2.COLOR_BGR2GRAY)

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    (ref_img_corners,reF_corner_count,ref_stddev) = get_corners(ref_img_gray,
                                ch_block_size,
                                ch_ksize,
                                ch_k,
                                ref_file_id,
                                output_file_extension,
                                out_path)
    ref_img_corners_coords = np.array(np.nonzero(ref_img_corners)).T


    """
    local maxima
    """
    min_stddev = ref_stddev

    ref_local_max_corners,ref_local_max_corners_coords = apply_corners_local_maxima(
                                ref_img_corners,
                                min_stddev,
                                local_maxima_stddev_threshold_factor,
                                ref_file_id,
                                output_file_extension,
                                out_path)

    draw_markers("embed_max_corners",
                ref_image,
                ref_local_max_corners_coords,
                color=[0,255,0],
                file_name=ref_file_id,
                output_file_extension=output_file_extension,
                path=out_path,
                display=False)

    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    ref_anms_coords = apply_anms_to_img(ref_local_max_corners_coords,
                                ref_local_max_corners,
                                max_features)
    print(f"ref:{len(ref_anms_coords)}")

    draw_markers("anms_corners",
            ref_image,
            ref_anms_coords,
            color=[0,255,0],
            file_name=ref_file_id,
            output_file_extension=output_file_extension,
            path=out_path,
            display=False)


    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    
    _, ref_fds = get_corner_descriptors(ref_img_gray, ref_local_max_corners_coords)

    if len(ref_fds) <= 0:
        print(ref_img_corners_coords)
        print("no feture_descriptors")
        return False,None

    return True,ref_fds

def get_img_correspondences(ref_image_orig, to_image_orig, ref_fds,to_fds,Args,ref_file_id,to_file_id):
    print(f"\n\n Running get_img_correspondences for {ref_file_id}<-{to_file_id}")
    ref_image = ref_image_orig.copy()
    to_image = to_image_orig.copy()
    ch_block_size = Args.CornerHarrisBlockSize
    ch_ksize = Args.CornerHarrisSobelOpSize
    ch_k = Args.CornerHarrisKParameter
    n_max_ransac_iterations = Args.RansacMaxIterations
    local_maxima_stddev_threshold_factor = Args.StdDevThresholdFactorForLocalMaxima
    max_features =  Args.ANMSMaxFeatures
    base_path = Args.BasePath
    img_set = Args.TestSet
    input_file_extension = '.jpg'
    output_file_extension = '.png'
    out_path = base_path + img_set

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    (ref_kps,to_kps,kps_dists) = get_feature_matches(ref_fds,to_fds)
    """
    Below portion of the code is to draw image correlations
    """
    draw_feature_matches(ref_kps,
            to_kps,
            kps_dists,
            ref_image,
            to_image,
            ref_file_id,
            to_file_id,
            output_file_extension,
            out_path)
    """
    Refine: RANSAC, Estimate Homography
    """
    warnings.filterwarnings("error")
    max_inliers = perform_ransac(n_max_ransac_iterations, ref_kps, to_kps)
    warnings.filterwarnings("always")

    """
    show output of ransac homography
    """
    draw_feature_matches(ref_kps[max_inliers],
                to_kps[max_inliers],
                kps_dists[max_inliers],
                ref_image,
                to_image,
                ref_file_id,
                to_file_id,
                output_file_extension,
                out_path,
                suffix="ransac")
    
    H = find_homography(ref_kps,to_kps,max_inliers)

    return True, H,np.sum(max_inliers)

def get_effective_homography(I,graph,reference_id,eff_homography_list,eff_homography_ids):
    visited = False
    print(f"I:{I}")
    if I == reference_id:
        return None
    for j in range(graph.shape[0]):
        if visited:
            print("something is wrong")
            return
        if j == reference_id:
            continue
        if graph[j,I] != 0:
            visited = True
            I_d = eff_homography_ids[I]
            val = get_effective_homography(j,graph,reference_id,eff_homography_list,eff_homography_ids)
            if val is None:
                continue
            eff_homography_list[I_d] = val*eff_homography_list[I_d]
    return eff_homography_list[eff_homography_ids[I]]

def n_node_conxns(i,graph):
    return len(np.nonzero(graph[i]))
def node_total_weights(i,graph):
    return graph[i].sum()

def stitch_images(ref_image,to_image,H,T,ref_file_id,to_file_id,output_file_extension,out_path):
    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """
    H = H + T
    tf_to_img,translation_vec,final_to_img_shape,tf_to_bb_coords = get_full_homography_img(to_image,H)
    write_image_output(f"image{ref_file_id}_{to_file_id}_perspective",
            to_file_id,
            output_file_extension,
            tf_to_img,
            out_path,
            display=False)
    
    tf_ref_img,tf_ref_bb_coords,final_canvas_shape = affine_and_resize_image(ref_image,translation_vec,final_to_img_shape)
    write_image_output(f"image{ref_file_id}_affined_{to_file_id}",
            ref_file_id,
            output_file_extension,
            tf_ref_img,
            out_path,
            display=False)

    stitch_img = np.zeros(shape=(final_canvas_shape[0],final_canvas_shape[1],3))
    stitch_img[0:final_to_img_shape[0],0:final_to_img_shape[1],:] = tf_to_img
    stitch_img = tf_ref_img
    print(stitch_img.shape)
    print(tf_to_img.shape)
    # write_image_output(f"image{ref_file_id}{to_file_id}_stitch",
    #         to_file_id,
    #         output_file_extension,
    #         stitch_img,
    #         out_path,
    #         display=False)

    print(f"to {tf_to_bb_coords[0:2].T}")
    print(f"ref {tf_ref_bb_coords[0:2].T}")

    poly1 = Polygon(map(Point, tf_to_bb_coords[0:2].T))
    poly2 = Polygon(map(Point, tf_ref_bb_coords[0:2].T))
    intersected_poly = poly1.intersection(poly2)
    
    for i in range(stitch_img.shape[0]):
        for j in range(stitch_img.shape[1]):
            if poly1.contains(Point(j,i)) and tf_to_img[i,j,0] !=0 and  tf_to_img[i,j,1] !=0 and tf_to_img[i,j,2] !=0:
                stitch_img[i,j] = tf_to_img[i,j]
    
    write_image_output(f"image{ref_file_id}{to_file_id}_clear_stitch",
        to_file_id,
        output_file_extension,
        stitch_img,
        out_path,
        display=False)
    affine_mat = np.zeros(shape=(3,3))
    affine_mat[0:2,2] = translation_vec.T[0,0:2]
    return stitch_img,f"{ref_file_id}{to_file_id}",affine_mat

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath',default='../Data/Train/',help='Base path to get images from,Default:../Data/Train/')
    Parser.add_argument('--TestSet',default='Set1/',help='Test set to run algorithm on,Default:Set1')
    Parser.add_argument('--CornerHarrisBlockSize', default=4, help='Block Size for Harris Corner Detection, Default:7')
    Parser.add_argument('--CornerHarrisSobelOpSize', default=7, help='Block Size for Harris Corner Detection, Default:11')
    Parser.add_argument('--CornerHarrisKParameter', default=0.04, help='Block Size for Harris Corner Detection, Default:0.04')
    Parser.add_argument('--ANMSMaxFeatures', default=10000, help='Block Size for Harris Corner Detection, Default:0.04')
    Parser.add_argument('--StdDevThresholdFactorForLocalMaxima',default=1,help='threshold for local maxima based on standard deviation of standardized corner score,Default:1')
    Parser.add_argument('--RansacMaxIterations',default=10000,help='Maximum number of iterations a RANSAC algorithm should run, Default:10000')

    Args = Parser.parse_args()
    base_path = Args.BasePath
    img_set = Args.TestSet
    input_file_extension = '.jpg'
    output_file_extension = '.png'
    
    """
    Read a set of images for Panorama stitching
    """
    print(base_path+img_set)
    files = glob.glob(base_path + img_set + "*" + input_file_extension,recursive=False)
    files.sort()
    files = [file.replace("\\","/") for file in files]
    print("List of files to read:",files)
    file_names = [file.replace(base_path+img_set,'').replace(input_file_extension,'') for file in files]
    images_color = []
    for image_file in files:
        img_color_orig = cv2.imread(image_file)
        images_color.append(img_color_orig)

    images_color = np.array(images_color)

    """
    Get all the needed information like feature descriptors
    """
    fds = []
    for i in range(len(files)):
        isGood, img_fds = get_image_info(images_color[i],Args,file_names[i])
        fds.append([isGood, img_fds])

    """
    get image correspondences of all possible pairs
    store them in adjacency matrix to construct a best possible graph
    """
    imgs_graph = np.zeros(shape=(len(files),len(files)))
    homography_mats_list = []
    homography_inds_mat = np.zeros(shape=(len(files),len(files)),dtype=int)
    for i in range(len(files)):
        if not fds[i][0]:
            continue
        for j in range(len(files)):
            if not fds[j][0]:
                continue
            if i == j:
                continue
            print(f"\n\nRunning Iteration for {i},{j}")
            isGood, H, inliers = get_img_correspondences(images_color[i],images_color[j],fds[i][1],fds[j][1],Args,file_names[i],file_names[j])
            if not isGood:
                continue
            print(f"inliers:{inliers}")
            imgs_graph[i,j] = inliers
            homography_mats_list.append(H)
            homography_inds_mat[i,j] = len(homography_mats_list) - 1

    print(imgs_graph)
    print(homography_inds_mat)
    # ref_image = images_color[0]
    # ref_file_id = file_names[0]
    # T = np.zeros(shape=(3,3))
    # for i in range(1,3):
    #     homography = homography_mats_list[homography_inds_mat[0,i]] + T
    #     ref_image,ref_file_id,T = stitch_images(ref_image,images_color[i],homography,T,ref_file_id,file_names[i],output_file_extension,base_path+img_set)
    # return
    """
    Recognizing panorama images
    """
    """
    Step: 1
        Thresholdinng the inliers
    """
    imgs_graph[imgs_graph < 10 ] = 0
    print(f"after thresholding:\n{imgs_graph}")
    """
    Step: 2
        Identify directions of the panorama graph
        from homography matrix
    """
    imgs_graph[imgs_graph < imgs_graph.T] = 0
    print(f"after fixing directions:\n{imgs_graph}")
    """
    Step: 3
        fix final directions by taking
        max features per column
    """
    imgs_graph[np.max(imgs_graph,axis=0,keepdims=True)!=imgs_graph]=0
    for i in range(imgs_graph.shape[0]):
        for j in range(imgs_graph.shape[0]):
            if imgs_graph[i,j] != 0 and imgs_graph[i,j] == imgs_graph[j,i]:
                if n_node_conxns(i,imgs_graph) > n_node_conxns(j,imgs_graph):
                    imgs_graph[j,i] = 0
                elif n_node_conxns(j,imgs_graph) > n_node_conxns(i,imgs_graph):
                    imgs_graph[i,j] = 0
                else:
                    if node_total_weights(i,imgs_graph) >= node_total_weights(j,imgs_graph):
                        imgs_graph[j,i] = 0
                    else:
                        imgs_graph[i,j] = 0

    print(f"after finalizing node connections:\n{imgs_graph}")
    """
    Step: 4
        get reference image
    """
    ref_image_id = np.where(~imgs_graph.any(axis=0))[0][0]
    print(f"ref_image_id:{ref_image_id}")
    """
    Step: 5
    find the effective homography transformation 
        for each image to the ref_image_id
        essentially ref_image_id is the root of the graph
        we need to calculate effective homography of every node 
            to the root node
        effective homography is just multiplication of 
            all the homographies
    """
    # initialize the effective homograpy matrix list
    eff_H_list = []
    eff_H_list_ids = np.full(imgs_graph.shape[0],dtype=int,fill_value=-1)
    nonzero_list_id = np.nonzero(imgs_graph)
    for k,j in enumerate(nonzero_list_id[1]):
        print(nonzero_list_id[0][k],j)
        eff_H = homography_mats_list[homography_inds_mat[nonzero_list_id[0][k],j]]
        eff_H_list.append(eff_H)
        eff_H_list_ids[j] = len(eff_H_list) - 1

    print(eff_H_list)
    print(eff_H_list_ids)

    # calculate the effective homography of every node to the root node
    eff_H_vals = np.full((imgs_graph.shape[0],3,3),fill_value = np.identity(3))
    for i in range(imgs_graph.shape[0]):
        print(f"i:{i}")
        val = get_effective_homography(i,imgs_graph,ref_image_id,eff_H_list,eff_H_list_ids)
        if val is None:
            continue
        eff_H_vals[i] = val
    print(eff_H_vals)

    ref_image = images_color[ref_image_id]
    ref_file_id = file_names[ref_image_id]
#    stitch_images(ref_image,images_color[0],eff_H_vals[0],ref_file_id,file_names[0],output_file_extension,base_path+img_set)
    T = np.zeros(shape=(3,3))

    for i in range(0,len(files)):
        if i == ref_image_id:
            continue
        ref_image,ref_file_id,T = stitch_images(ref_image,images_color[i],eff_H_vals[i],T,ref_file_id,file_names[i],output_file_extension,base_path+img_set)
        print(i,ref_file_id)
    return


if __name__ == "__main__":
    main()
print("end of file")
