import os
import sys
import subprocess 
import argparse
import time
import math
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from cv2 import *
from scipy import ndimage
import scipy.signal
from scipy.spatial.distance import directed_hausdorff
from skimage import feature
from skimage.feature import match_template
from skimage.filters import threshold_otsu
from skimage.transform import rescale
import imutils
from joblib import Parallel, effective_n_jobs, delayed
from sklearn.utils import gen_even_slices
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path
from shutil import copyfile
from helper_functions import save_obj, get_image_2d_matrix, get_particle_count, get_particle_count_dict_cluster, remove_files_in_folder 
from extract_relion_particle_counts import get_particle_counts


def crop_image(img):    
    row_idx, col_idx = np.nonzero(img)
    return(img[np.min(row_idx):np.max(row_idx)+1,np.min(col_idx):np.max(col_idx)+1])


def norm_cross_correlation_cv(img1, img2):
      
    cross_image = cv2.filter2D(img1, -1, img2, borderType=cv2.BORDER_CONSTANT)
     
    max_idx = np.unravel_index(np.argmax(cross_image), cross_image.shape)
     
    relative_diff_y = max_idx[0] - img1.shape[0]//2
    relative_diff_x = max_idx[1] - img1.shape[1]//2
     
    return(cross_image, max_idx, relative_diff_x, relative_diff_y)


def get_image_rotation_matrix(image_2d_matrix, relevant_image_list, scale_factor, mirror):

    #each image has a unique maximum shape  
    image_2d_rotation_matrix_map = {}
    image_2d_rotation_matrix_max_shape_map = {}

    image_2d_rotation_mirror_matrix_map = {}
    image_2d_rotation_mirror_matrix_max_shape_map = {}

    for i in relevant_image_list:

        curr_img = np.copy(image_2d_matrix[i,:,:])

        if scale_factor is not None:
            curr_img = rescale(curr_img, scale_factor, anti_aliasing=True)

        max_height, max_width, rotation_matrix_map = get_rotated_image_max_shape(curr_img)
        image_2d_rotation_matrix_map[i] = rotation_matrix_map
        image_2d_rotation_matrix_max_shape_map[i] = [max_height, max_width]
        
        if mirror:
            curr_img = np.flip(curr_img, axis=0)
            max_height, max_width, rotation_matrix_map = get_rotated_image_max_shape(curr_img)
            image_2d_rotation_mirror_matrix_map[i] = rotation_matrix_map
            image_2d_rotation_mirror_matrix_max_shape_map[i] = [max_height, max_width] 

    rotation_matrix_map = {}
    max_shape_map = {}

    rotation_matrix_map['original'] = image_2d_rotation_matrix_map
    max_shape_map['original'] = image_2d_rotation_matrix_max_shape_map

    if mirror:
        rotation_matrix_map['mirror'] = image_2d_rotation_mirror_matrix_map
        max_shape_map['mirror'] = image_2d_rotation_mirror_matrix_max_shape_map

                     
    return(rotation_matrix_map, max_shape_map)           
       
 
def get_rotated_image_max_shape(img):

    rotation_angles = range(0,360,6)

    max_shape = np.array([0,0])

    max_idx_row = -1
    max_idx_col = -1

    curr_img = np.copy(img)

    rotation_matrix_map = {}
    
    for j in rotation_angles:

        rotated_img = imutils.rotate_bound(curr_img, j)
        rotated_img_cropped = crop_image(rotated_img)
        rotated_img_cropped_shape = rotated_img_cropped.shape

        if rotated_img_cropped_shape[0] > max_shape[0]:
            max_shape[0] = rotated_img_cropped_shape[0]
            max_idx_row = j

        if rotated_img_cropped_shape[1] > max_shape[1]:
            max_shape[1] = rotated_img_cropped_shape[1]
            max_idx_col = j 
        
        rotation_matrix_map[j] = rotated_img_cropped

    return((max_shape[0], max_shape[1], rotation_matrix_map))


def rot_trans_invariant_dist_optimized(image_2d_matrix, img1_idx, img2_idx, scale_factor, mirror_bool, mrc_height, mrc_width):
    
    rotation_angles = range(0,360,6)

    image_2d_rotation_matrix_map, image_2d_rotation_matrix_max_shape_map = get_image_rotation_matrix(image_2d_matrix, [img1_idx, img2_idx], scale_factor, mirror_bool)
    
    img1_cropped = image_2d_rotation_matrix_map['original'][img1_idx][0]
    
    max_height = img1_cropped.shape[0]
    max_width = img1_cropped.shape[1]
    
    if mirror_bool:
        img2_max_height = image_2d_rotation_matrix_max_shape_map['mirror'][img2_idx][0]
        img2_max_width = image_2d_rotation_matrix_max_shape_map['mirror'][img2_idx][1]
    else:
        img2_max_height = image_2d_rotation_matrix_max_shape_map['original'][img2_idx][0]
        img2_max_width = image_2d_rotation_matrix_max_shape_map['original'][img2_idx][1]

 
    if img2_max_height > max_height:
        max_height = img2_max_height
    if img2_max_width > max_width:
        max_width = img2_max_width

    ###img1 and img2 are mapped to the same dimensions -- note that comparisons between different sets of images are not of the same size###
    ###normalized cross correlation is independent of size (in the zero-padded sense) but hausdroff distance is not###
    
    img1_cropped_padded = np.zeros((max_height,max_width))

    # compute offset
    x_start_new = (max_width - img1_cropped.shape[1]) // 2
    y_start_new = (max_height - img1_cropped.shape[0]) // 2

    # copy image into center of result image
    img1_cropped_padded[y_start_new:y_start_new+img1_cropped.shape[0], 
                        x_start_new:x_start_new+img1_cropped.shape[1]] = img1_cropped 
    
    img2_rotation_2d_matrix = np.zeros((len(rotation_angles), max_height, max_width))
    
    for i,angle in enumerate(rotation_angles):
       
        if mirror_bool: 
            rotated_img2_cropped = image_2d_rotation_matrix_map['mirror'][img2_idx][angle]    
        else:
            rotated_img2_cropped = image_2d_rotation_matrix_map['original'][img2_idx][angle]    
        
        padded_output = np.zeros((max_height,max_width))

        # compute  offset
        x_start_new = (max_width - rotated_img2_cropped.shape[1]) // 2
        y_start_new = (max_height - rotated_img2_cropped.shape[0]) // 2
        
        # copy image into center of result image
        padded_output[y_start_new:y_start_new+rotated_img2_cropped.shape[0], 
                       x_start_new:x_start_new+rotated_img2_cropped.shape[1]] = rotated_img2_cropped 
        
        img2_rotation_2d_matrix[i,:,:] = padded_output
        
    correlation_dist_matrix = np.zeros(len(rotation_angles))
    correlation_params = np.zeros((len(rotation_angles), 3)) #rotation angle, relative_diff_y, relative_diff_x


    img1_cropped_padded -= np.mean(img1_cropped_padded)
    img2_rotation_2d_matrix_mean = np.mean(img2_rotation_2d_matrix, axis=(1,2))
    img2_rotation_2d_matrix = img2_rotation_2d_matrix - img2_rotation_2d_matrix_mean.reshape(img2_rotation_2d_matrix_mean.shape[0],1,1)
        
    img1_cropped_padded /= np.std(img1_cropped_padded)
    img2_rotation_2d_matrix_std = np.std(img2_rotation_2d_matrix, axis=(1,2))
    img2_rotation_2d_matrix = img2_rotation_2d_matrix/img2_rotation_2d_matrix_std.reshape(img2_rotation_2d_matrix_std.shape[0],1,1)
    
    img1_cropped_padded_length_norm = img1_cropped_padded/(img1_cropped_padded.shape[0]*img1_cropped_padded.shape[1])        
    
    for i in range(0,len(rotation_angles)):
        
        cross_image_mat, max_idx, relative_diff_x, relative_diff_y = norm_cross_correlation_cv(img1_cropped_padded_length_norm, img2_rotation_2d_matrix[i,:,:]) #shift of img2 wrt img1
        max_cross_cor = cross_image_mat[max_idx]
        correlation_dist_matrix[i] = max_cross_cor
            
        '''img2_shifted = ndimage.shift(np.copy(img2_rotation_2d_matrix[i,:,:]), (relative_diff_y, relative_diff_x))
        if blur:
            cross_corr_match_template = match_template(cv2.GaussianBlur(img2_shifted,(51,51),0), cv2.GaussianBlur(np.copy(img1_cropped_padded),(51,51),0))
        else:
            cross_corr_match_template = match_template(img2_shifted, img1_cropped_padded)
        print("This %f and this %f should roughly match" % (max_cross_cor, cross_corr_match_template[0][0]))
        print("relative y %d, relative x %d" % (relative_diff_y, relative_diff_x))'''
        
                
        correlation_params[i,0] = rotation_angles[i]
        correlation_params[i,1] = relative_diff_y
        correlation_params[i,2] = relative_diff_x

    max_corr_idx = np.argmax(correlation_dist_matrix)   
    angle_optimal = correlation_params[max_corr_idx,0] 
    relative_diff_y_optimal = correlation_params[max_corr_idx,1]
    relative_diff_x_optimal = correlation_params[max_corr_idx,2] 

    return(np.max(correlation_dist_matrix), angle_optimal, relative_diff_y_optimal, relative_diff_x_optimal)
    

def parallel_pairwise_dist_matrix(image_2d_matrix, scale_factor, mirror_indicator_matrix, mrc_height, mrc_width, dist_wrapper, dist_func, unique_relevant_image_pairs, n_jobs):
    """Break the pairwise matrix in n_jobs even slices
    and compute them in parallel."""

    # enforce a threading backend to prevent data communication overhead
    fd = delayed(dist_wrapper)

    num_images_all = image_2d_matrix.shape[0]
    corr_ret = np.zeros((num_images_all, num_images_all))
    edge_ret = np.zeros((num_images_all, num_images_all))
    rot_angle_ret = np.zeros((num_images_all, num_images_all))
    ytrans_ret = np.zeros((num_images_all, num_images_all))
    xtrans_ret = np.zeros((num_images_all, num_images_all))
    mirror_ret = np.zeros((num_images_all, num_images_all))

    result = Parallel(backend="loky", n_jobs=n_jobs, verbose=10)(
        fd(unique_relevant_image_pairs[s], image_2d_matrix, scale_factor, mirror_indicator_matrix, mrc_height, mrc_width, dist_func)
        for s in gen_even_slices(len(unique_relevant_image_pairs), effective_n_jobs(n_jobs)))


    for i in range(0,len(result)):

        corr_dist_list = result[i][0] 
        rot_angle_list = result[i][1] 
        ytrans_list = result[i][2] 
        xtrans_list = result[i][3] 
        mirror_indicator_list = result[i][4] 
     
        for j in range(0,len(corr_dist_list)):

            idx1 = corr_dist_list[j][0]
            idx2 = corr_dist_list[j][1]

            corr_ret[idx1,idx2] = corr_dist_list[j][2]
            rot_angle_ret[idx1,idx2] = rot_angle_list[j][2] 
            ytrans_ret[idx1,idx2] = ytrans_list[j][2]
            xtrans_ret[idx1,idx2] = xtrans_list[j][2]
            mirror_ret[idx1,idx2] = mirror_indicator_list[j][2]

    
    return((corr_ret, rot_angle_ret, ytrans_ret, xtrans_ret, mirror_ret))


def rot_trans_invariant_dist_wrapper(unique_relevant_image_pairs, image_2d_matrix, scale_factor, mirror_indicator_matrix, mrc_height, mrc_width, dist_func):
 
    corr_dist_list = []
    rot_angle_list = []
    ytrans_list = []
    xtrans_list = []
    mirror_indicator_list = []
 
    for i in range(0,len(unique_relevant_image_pairs)):

        img1_idx = unique_relevant_image_pairs[i][0]
        img2_idx = unique_relevant_image_pairs[i][1]
        mirror = mirror_indicator_matrix[np.min([img1_idx, img2_idx]), np.max([img1_idx, img2_idx])] #upper triangular only
        correlation_dist, angle_optimal, relative_diff_y_optimal, relative_diff_x_optimal = dist_func(image_2d_matrix, img1_idx, img2_idx, scale_factor, mirror, mrc_height, mrc_width)

        corr_dist_list.append([img1_idx, img2_idx, correlation_dist])
        rot_angle_list.append([img1_idx, img2_idx, angle_optimal])
        ytrans_list.append([img1_idx, img2_idx, relative_diff_y_optimal])
        xtrans_list.append([img1_idx, img2_idx, relative_diff_x_optimal])
        mirror_indicator_list.append([img1_idx, img2_idx, mirror])
 
    return((corr_dist_list, rot_angle_list, ytrans_list, xtrans_list, mirror_indicator_list))




def get_missing_alignment_parameters(image_list_cluster_c, alignment_parameters, community_image_list, ref_image_list, input_dir):

    relevant_image_pairs = [] 

    for i in range(0,len(community_image_list)):

        for j in range(0,len(community_image_list[i])):
        
            curr_community_image_list = community_image_list[i][j]
            curr_ref_image = ref_image_list[i][j]
                
            curr_community_original_image_list = [image_list_cluster_c[x] for x in curr_community_image_list]
            curr_ref_original_image = image_list_cluster_c[curr_ref_image]
            
            for k in range(0,len(curr_community_original_image_list)):
                
                if curr_ref_original_image > curr_community_original_image_list[k]:
                
                     relevant_image_pairs.append([curr_ref_original_image, curr_community_original_image_list[k]])

    unique_relevant_image_pairs = np.unique(np.array(relevant_image_pairs), axis=0) #list of lists
    unique_relevant_images = np.unique(np.array(relevant_image_pairs))

    print(unique_relevant_image_pairs) 
     
    image_2d_matrix = get_image_2d_matrix(input_dir)
    mrc_height = image_2d_matrix.shape[1]
    mrc_width = image_2d_matrix.shape[2]
 
    scale_factor = float(input_dir.split('_')[-1].split('=')[-1])
    mirror_indicator_matrix = np.genfromtxt('%s/pairwise_matrix/mirror_indicator_matrix.csv' % input_dir, delimiter=',')

    print('calculating distance matrix for missing parameters')
    corr_dist_matrix, rot_angle_matrix, ytrans_matrix, xtrans_matrix, mirror_indicator_matrix = parallel_pairwise_dist_matrix(image_2d_matrix, scale_factor, mirror_indicator_matrix, mrc_height, mrc_width, rot_trans_invariant_dist_wrapper, rot_trans_invariant_dist_optimized, unique_relevant_image_pairs, -1)

    return(rot_angle_matrix, xtrans_matrix, ytrans_matrix, mirror_indicator_matrix)


