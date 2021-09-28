import os
import sys
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
from helper_functions import load_obj, save_obj, sort_dict 
from extract_relion_particle_counts import get_particle_counts


def remove_neg_values(img):
    img[img < 0] = 0
    return(img)


def crop_image_cv(img):
    img_s = cv2.convertScaleAbs(img)
    x, y, w, h = cv2.boundingRect(img_s)
    tol = 10
    xmin_tol = np.max([0, x-tol])
    ymin_tol = np.max([0, y-tol])
    xmax_tol = np.min([img.shape[1],x+w+tol])
    ymax_tol = np.min([img.shape[0],y+h+tol])
    
    return(img[ymin_tol:ymax_tol, xmin_tol:xmax_tol])


def crop_image(img):    
    row_idx, col_idx = np.nonzero(img)
    return(img[np.min(row_idx):np.max(row_idx)+1,np.min(col_idx):np.max(col_idx)+1])



def norm_cross_correlation_cv(img1, img2):
      
    cross_image = cv2.filter2D(img1, -1, img2, borderType=cv2.BORDER_CONSTANT)
     
    max_idx = np.unravel_index(np.argmax(cross_image), cross_image.shape)
     
    relative_diff_y = max_idx[0] - img1.shape[0]//2
    relative_diff_x = max_idx[1] - img1.shape[1]//2
     
    return(cross_image, max_idx, relative_diff_x, relative_diff_y)


def get_image_rotation_matrix(image_2d_matrix, scale_factor, mirror):

    #each image has a unique maximum shape  
    image_2d_rotation_matrix_map = {}
    image_2d_rotation_matrix_max_shape_map = {}

    image_2d_rotation_mirror_matrix_map = {}
    image_2d_rotation_mirror_matrix_max_shape_map = {}

    for i in range(0,image_2d_matrix.shape[0]):

        #curr_img = remove_neg_values(np.copy(image_2d_matrix[i,:,:]))
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

    '''for i in range(0,image_2d_matrix.shape[0]):

        if mirror == 1:

            norm_cross_corr = rot_trans_invariant_dist_optimized(rotation_matrix_map, max_shape_map, i, i, True, image_2d_matrix.shape[1], image_2d_matrix.shape[2], True)
            if norm_cross_corr < .9:
                image_symmetry_indicator_map[i] = 0
            else:
                image_symmetry_indicator_map[i] = 1'''
                     
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


def rot_trans_invariant_dist_optimized(image_2d_rotation_matrix_map, image_2d_rotation_matrix_max_shape_map, img1_idx, img2_idx, mirror_bool, mrc_height, mrc_width, corr_only=False):
    
    rotation_angles = range(0,360,6)
    
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

    if corr_only:
        return(np.max(correlation_dist_matrix))

    max_corr_idx = np.argmax(correlation_dist_matrix)   
    angle_optimal = correlation_params[max_corr_idx,0] 
    relative_diff_y_optimal = correlation_params[max_corr_idx,1]
    relative_diff_x_optimal = correlation_params[max_corr_idx,2] 
    
    zero_val_mapping = img2_rotation_2d_matrix_mean[max_corr_idx]/img2_rotation_2d_matrix_std[max_corr_idx]
    img2_shifted = ndimage.shift(np.copy(img2_rotation_2d_matrix[max_corr_idx,:,:]), (relative_diff_y_optimal, relative_diff_x_optimal), cval = 0-zero_val_mapping) 

    #to ignore background noise when calculating edges -- 2 rounds of otsu thresholding
    img1_th = (float(threshold_otsu(cv2.GaussianBlur(img1_cropped_padded,(21,21),0))))
    img2_th = (float(threshold_otsu(cv2.GaussianBlur(img2_shifted,(21,21),0))))
    
    img2_shifted[img2_shifted < img2_th] = 0
    img1_cropped_padded[img1_cropped_padded < img1_th] = 0
    
    img1_th = (float(threshold_otsu(img1_cropped_padded)))
    img2_th = (float(threshold_otsu(img2_shifted)))
    
    img2_shifted[img2_shifted < img2_th] = 0
    img1_cropped_padded[img1_cropped_padded < img1_th] = 0

     
    ##calculate distance between edges using hausdroff metric looping thorugh different values of sigma
    
    sigma_vals = [1,2]
    hausdorff_norm = math.sqrt(math.pow(mrc_height,2) + math.pow(mrc_width,2)) 
    
    hausdroff_dist_matrix = np.zeros((len(sigma_vals), len(sigma_vals)))

    img2_edge_matrix = np.zeros((len(sigma_vals), img2_shifted.shape[0], img2_shifted.shape[1]))
    
    for i,s1 in enumerate(sigma_vals):
        
        img2_edge = feature.canny(img2_shifted, sigma=s1).astype(float)
        img2_edge_matrix[i,:,:] = img2_edge


    for i,s1 in enumerate(sigma_vals):
        
        img1_edge = feature.canny(img1_cropped_padded, sigma=s1).astype(float)
        
        for j,s2 in enumerate(sigma_vals):
            
            img2_edge = img2_edge_matrix[j,:,:]

            img1_edge_idx = np.argwhere(img1_edge == 1) 
            img2_edge_idx = np.argwhere(img2_edge == 1) 
         
            #hausdroff_21 = directed_hausdorff(img2_edge_idx, img1_edge_idx)[0]
            #hausdroff_12 = directed_hausdorff(img1_edge_idx, img2_edge_idx)[0]

            hausdroff_21 = (np.quantile(euclidean_distances(img2_edge_idx, img1_edge_idx).min(axis = 0), .95, axis=0))
            hausdroff_12 = (np.quantile(euclidean_distances(img2_edge_idx, img1_edge_idx).min(axis = 1), .95, axis=0))

            max_hausdroff_dist = np.max([hausdroff_21, hausdroff_12])

            #at certain values of sigma, no edges may be detected leading to a 0 hausdroff distance; we should ignore these 
            #if two images are identical, hausdorff matrix will be all np.inf which is equivalent to all zeros 
            if max_hausdroff_dist == 0:
                if len(img1_edge_idx) == 0 and len(img2_edge_idx) == 0:
                    max_hausdroff_dist = np.inf
                hausdroff_dist_matrix[i,j] = max_hausdroff_dist
            else:
                hausdroff_dist_matrix[i,j] = max_hausdroff_dist/hausdorff_norm           

        
    return(np.max(correlation_dist_matrix), np.min(hausdroff_dist_matrix), angle_optimal, relative_diff_y_optimal, relative_diff_x_optimal)


def dist_write(slice_, dist_wrapper, dist_func, corr_dist_matrix, edge_dist_matrix, rot_angle_matrix, ytrans_matrix, xtrans_matrix, mirror_indicator_matrix, rotation_matrix_map, max_shape_map, image_2d_matrix, mirror, mrc_height, mrc_width):
    """Write in-place to a slice of a distance matrix."""
    corr_dist_matrix_slice, edge_dist_matrix_slice, angle_slice, y_slice, x_slice, mirror_slice = dist_wrapper(slice_, rotation_matrix_map, max_shape_map, image_2d_matrix, mirror, mrc_height, mrc_width, dist_func)

    corr_dist_matrix[slice_,:] = corr_dist_matrix_slice
    edge_dist_matrix[slice_,:] = edge_dist_matrix_slice
    
    rot_angle_matrix[slice_,:] = angle_slice
    ytrans_matrix[slice_,:] = y_slice 
    xtrans_matrix[slice_,:] = x_slice 
    mirror_indicator_matrix[slice_,:] = mirror_slice
    

def parallel_pairwise_dist_matrix(rotation_matrix_map, max_shape_map, image_2d_matrix, mirror, mrc_height, mrc_width, dist_wrapper, dist_func, n_jobs):
    """Break the pairwise matrix in n_jobs even slices
    and compute them in parallel."""

    # enforce a threading backend to prevent data communication overhead
    fd = delayed(dist_write)
    corr_ret = np.zeros((image_2d_matrix.shape[0], image_2d_matrix.shape[0]))
    edge_ret = np.zeros((image_2d_matrix.shape[0], image_2d_matrix.shape[0]))
    rot_angle_ret = np.zeros((image_2d_matrix.shape[0], image_2d_matrix.shape[0]))
    ytrans_ret = np.zeros((image_2d_matrix.shape[0], image_2d_matrix.shape[0]))
    xtrans_ret = np.zeros((image_2d_matrix.shape[0], image_2d_matrix.shape[0]))
    mirror_ret = np.zeros((image_2d_matrix.shape[0], image_2d_matrix.shape[0]))

    Parallel(backend="threading", n_jobs=n_jobs)(
        fd(s, dist_wrapper, dist_func, corr_ret, edge_ret, rot_angle_ret, ytrans_ret, xtrans_ret, mirror_ret, rotation_matrix_map, max_shape_map, image_2d_matrix, mirror, mrc_height, mrc_width)
        for s in gen_even_slices(image_2d_matrix.shape[0], effective_n_jobs(n_jobs)))
    
    return((corr_ret, edge_ret, rot_angle_ret, ytrans_ret, xtrans_ret, mirror_ret))


def rot_trans_invariant_dist_wrapper(slice_, rotation_matrix_map, max_shape_map, image_2d_matrix, mirror, mrc_height, mrc_width, dist_func):
 
    num_images_subset = (slice_.stop - slice_.start)
    num_images_all = image_2d_matrix.shape[0]
    corr_dist_matrix = np.zeros((num_images_subset, num_images_all))
    edge_dist_matrix = np.zeros((num_images_subset, num_images_all))
    rot_angle_matrix = np.zeros((num_images_subset, num_images_all))
    ytrans_matrix = np.zeros((num_images_subset, num_images_all))
    xtrans_matrix = np.zeros((num_images_subset, num_images_all))
    mirror_indicator_matrix = np.zeros((num_images_subset, num_images_all))
   
    image_subset_indices = range(slice_.start, slice_.stop)
    
    if image_subset_indices[-1] == num_images_all:
        image_subset_indices = image_subset_indices[0:-1]
    
    for idx_i,i in enumerate(image_subset_indices):
        for idx_j,j in enumerate(range(i+1,num_images_all)):

            correlation_dist_orig, hausdroff_dist_orig, angle_optimal_orig, relative_diff_y_optimal_orig, relative_diff_x_optimal_orig = dist_func(rotation_matrix_map, max_shape_map, i, j, False, mrc_height, mrc_width)
 
            if mirror:
                correlation_dist_mirror, hausdroff_dist_mirror, angle_optimal_mirror, relative_diff_y_optimal_mirror, relative_diff_x_optimal_mirror = dist_func(rotation_matrix_map, max_shape_map, i, j, True, mrc_height, mrc_width)
            else:
                correlation_dist_mirror = -1

            if correlation_dist_orig >= correlation_dist_mirror:
                corr_dist_matrix[idx_i,j] = correlation_dist_orig #j and not idx_j because of size of corr_dist_matrix
                edge_dist_matrix[idx_i,j] = hausdroff_dist_orig
                rot_angle_matrix[idx_i,j] = angle_optimal_orig
                ytrans_matrix[idx_i,j] = relative_diff_y_optimal_orig
                xtrans_matrix[idx_i,j] = relative_diff_x_optimal_orig
                mirror_indicator_matrix[idx_i,j] = 0
            else:
                corr_dist_matrix[idx_i,j] = correlation_dist_mirror #j and not idx_j because of size of corr_dist_matrix
                edge_dist_matrix[idx_i,j] = hausdroff_dist_mirror
                rot_angle_matrix[idx_i,j] = angle_optimal_mirror
                ytrans_matrix[idx_i,j] = relative_diff_y_optimal_mirror
                xtrans_matrix[idx_i,j] = relative_diff_x_optimal_mirror
                mirror_indicator_matrix[idx_i,j] = 1
        
    return((corr_dist_matrix, edge_dist_matrix, rot_angle_matrix, ytrans_matrix, xtrans_matrix, mirror_indicator_matrix))


def convert_upper_triang_mat_to_symmetric(mat, matrix_type):
    
    if matrix_type == 'corr':
        diag_val = 1
    else:
        diag_val = 0

    tmp = np.copy(mat)
    for i in range(0,tmp.shape[1]):
        for j in range(i,tmp.shape[0]):
            if i == j:
                tmp[i,j] = diag_val
            else:
                tmp[j,i] = tmp[i,j]
    return(tmp)


def save_matrix(dist_matrix, matrix_type, output_dir, start_time):

    output_dir = '%s/pairwise_matrix' % output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    symm_dist_matrix = convert_upper_triang_mat_to_symmetric(dist_matrix, matrix_type)
    
    if matrix_type == 'corr':
        symm_dist_matrix = 1-symm_dist_matrix

    if matrix_type in ['corr', 'edge']:
        np.savetxt('%s/%s_dist_matrix.csv' % (output_dir, matrix_type), symm_dist_matrix, delimiter=',')    
    else:
        np.savetxt('%s/%s_matrix.csv' % (output_dir, matrix_type), symm_dist_matrix, delimiter=',') 
    np.savetxt('%s/execution_time.csv' % output_dir, np.array([(time.monotonic() - start_time)/60])) 

def gen_clean_input(mrc_file, particle_count_file, output_dir):

    output_dir = '%s/input' % output_dir 
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    mrc = mrcfile.open(mrc_file, mode='r')
    image_2d_matrix = mrc.data
    print('mrc shape before cleaning:')
    print(image_2d_matrix.shape)

    particle_count_map = load_obj(particle_count_file) 

    relevant_idx = []
    blank_image_exists = False    

    for i in range(0,image_2d_matrix.shape[0]):

        curr_img = image_2d_matrix[i,:,:]
        all_vals_same_bool = np.all(curr_img == curr_img[0])
        if not(all_vals_same_bool):
            relevant_idx.append(i)
        else:
            blank_image_exists = True

    image_2d_matrix_clean = image_2d_matrix[np.array(relevant_idx),:,:]        

    print('mrc shape after cleaning:')
    print(image_2d_matrix_clean.shape)

    if blank_image_exists:

        mrc_cleanfile = mrc_file.replace('.mrc','_clean.mrc')
        file_savepath = '%s/%s' % (output_dir, mrc_cleanfile.split('/')[-1])
        print('saving %s' % file_savepath)
        with mrcfile.new(file_savepath, overwrite=True) as mrc:
            mrc.set_data(image_2d_matrix_clean)

        particle_count_cleanfile = particle_count_file.replace('.pkl', '_clean.pkl')
        file_savepath = '%s/%s' % (output_dir, particle_count_cleanfile.split('/')[-1])
        print('saving %s' % file_savepath)
        particle_count_map_clean = {}
        for idx,i in enumerate(relevant_idx):
            particle_count_map_clean[idx] = particle_count_map[i]
        save_obj(particle_count_map_clean, file_savepath)

    else:

        copyfile(mrc_file, '%s/%s' % (output_dir, mrc_file.split('/')[-1]))
        copyfile(particle_count_file, '%s/%s' % (output_dir, particle_count_file.split('/')[-1]))

    return(image_2d_matrix_clean)


'''filename = 'cryosparc_P127_J4_020_class_averages_opt.mrc'
#filename = 'cryosparc_P50_J436_020_class_averages.mrc'

for filename in ['KaiC_class_averages.mrc', 'cryosparc_P50_J436_020_class_averages.mrc', 'hLonOpenClosed_class_averages.mrc', 'cryosparc_P127_J4_020_class_averages.mrc']:
#for filename in ['KaiC_class_averages.mrc']:

    print(filename)

    filename_wo_type = filename.split('.')[0]
    mrc_path = './raw_data'
    mrc_file = '%s/%s.mrc' % (mrc_path,filename_wo_type)

    image_2d_matrix = get_clean_image_2d_matrix(mrc_file)

    #print(image_2d_matrix.shape)

    start_time = time.monotonic()
    blur = False
    print('calculating rotated versions of images')
    image_2d_rotation_matrix_map, image_2d_rotation_matrix_max_shape_map = get_image_rotation_matrix(image_2d_matrix)
    print('calculating pairwise dist matrix')
    corr_dist_matrix, edge_dist_matrix, rot_angle_matrix, ytrans_matrix, xtrans_matrix = parallel_pairwise_dist_matrix(image_2d_rotation_matrix_map, image_2d_rotation_matrix_max_shape_map, image_2d_matrix, rot_trans_invariant_dist_wrapper, rot_trans_invariant_dist_optimized, blur, -1)
    print('saving dist matrix - clean')

    save_matrix(corr_dist_matrix, 'corr', filename_wo_type, blur, start_time)
    save_matrix(edge_dist_matrix, 'haus', filename_wo_type, blur, start_time)
    save_matrix(rot_angle_matrix, 'rot', filename_wo_type, blur, start_time)
    save_matrix(ytrans_matrix, 'ytrans', filename_wo_type, blur, start_time)
    save_matrix(xtrans_matrix, 'xtrans', filename_wo_type, blur, start_time)'''


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mrc_file", help="Location of *.mrc file", required=True)
    parser.add_argument("--star_file", help="Location of *.star file [metadata generated by RELION or CryoSPARC .cs file converted to .star via pyem (https://github.com/asarnow/pyem)]", required=True)
    parser.add_argument("--scale_factor", help="Factor [between (0,1)] by which to downsample images (defaults to 1 aka no downsampling)", type=float) 
    parser.add_argument("--mirror", help="Whether or not to run calculations for original class average and its mirror image (defaults to 1)", type=int, default=1)  
    parser.add_argument("--output_dir", help="Directory to save output files (defaults to directory where xyz.mrc file is stored and folder is called xyz_summary_mirror=0/1_scale=num)")
    args = parser.parse_args()

    if '.mrc' not in args.mrc_file:
        sys.exit('mrc_file must have extension .mrc')
    if '.star' not in args.star_file:
        sys.exit('star_file must have extensions .star')

    if args.scale_factor is None:
        scale_factor_str = '1'
    else:
        if args.scale_factor > 1 or args.scale_factor < 0:
            sys.exit('scale factor must be between 0 and 1 (exclusive)')
        scale_factor_str = str(args.scale_factor)

    mirror_str = str(int(args.mirror))

    if args.output_dir is None:
        output_dir = '%s_summary_mirror=%s_scale=%s' % ((args.mrc_file).split('.mrc')[0], mirror_str, scale_factor_str)
    else:
        output_dir = '%s_mirror=%s_scale=%s' % (args.output_dir, mirror_str, scale_factor_str)

    #save filepath.txt -- used for plotly visualization
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir_abs_path = str(Path(output_dir).absolute())
    output_dir_abs_path_norm = os.path.normpath(output_dir_abs_path)
    filepath_txt_file = '%s/filepath.txt' % output_dir
    with open(filepath_txt_file, "w") as text_file:
        text_file.write(output_dir_abs_path_norm)

    particle_count_file = get_particle_counts(args.star_file) 
    
    image_2d_matrix = gen_clean_input(args.mrc_file, particle_count_file, output_dir)
    mrc_height = image_2d_matrix.shape[1]
    mrc_width = image_2d_matrix.shape[2]
 
    start_time = time.monotonic()
    print('calculating rotated versions of images')
    rotation_matrix_map, max_shape_map = get_image_rotation_matrix(image_2d_matrix, args.scale_factor, args.mirror)
    print('calculating pairwise dist matrix')
    corr_dist_matrix, edge_dist_matrix, rot_angle_matrix, ytrans_matrix, xtrans_matrix, mirror_indicator_matrix = parallel_pairwise_dist_matrix(rotation_matrix_map, max_shape_map, image_2d_matrix, args.mirror, mrc_height, mrc_width, rot_trans_invariant_dist_wrapper, rot_trans_invariant_dist_optimized, -1)
    print('saving dist matrix - clean')

    save_matrix(corr_dist_matrix, 'corr', output_dir, start_time)
    save_matrix(edge_dist_matrix, 'edge', output_dir, start_time)
    save_matrix(rot_angle_matrix, 'rot', output_dir, start_time)
    save_matrix(ytrans_matrix, 'ytrans', output_dir, start_time)
    save_matrix(xtrans_matrix, 'xtrans', output_dir, start_time)
    save_matrix(mirror_indicator_matrix, 'mirror_indicator', output_dir, start_time)



