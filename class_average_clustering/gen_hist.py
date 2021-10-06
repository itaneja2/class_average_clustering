import sys
import os 
import argparse 
import numpy as np
import mrcfile
from joblib import Parallel, effective_n_jobs, delayed
from scipy import ndimage
from sklearn import cluster
from sklearn import metrics
import math
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
from sklearn.preprocessing import minmax_scale
from pathlib import Path
from cv2 import *
from helper_functions import save_obj, get_image_2d_matrix, get_particle_count, get_particle_count_dict_cluster, remove_files_in_folder 
import ref_image_alignment_parameters 
import imutils
 
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib.pyplot as plt
import seaborn as sns

def get_ref_image(dist_matrix_cluster_c, community_image_list, particle_count_dict_cluster_c):
     
    #get image with closest median distance to all other images 
    median_dist_list = []
    particle_count_list = []
    for i in community_image_list:
        dist_wrt_i = []
        particle_count_list.append(particle_count_dict_cluster_c[i])
        for j in community_image_list:
            dist_ij = dist_matrix_cluster_c[i,j]
            dist_wrt_i.append(dist_ij)
        dist_wrt_i = np.array(dist_wrt_i)
        median_dist_list.append(np.median(dist_wrt_i))
    
    if len(community_image_list) > 2:
        median_dist_list = np.array(median_dist_list)
        ref_img = community_image_list[np.argmin(median_dist_list)]
    else:
        ref_img = community_image_list[np.argmax(np.array(particle_count_list))]
    
    return(ref_img)


def get_average_image_wrt_ref(all_ref_image_list, all_community_image_list, particle_count_dict_cluster_c, image_list_cluster_c, alignment_parameters, input_dir):


    def apply_transformations(img, x, y, angle, mirror, max_height, max_width):

        #print('x=%d' % int(x))
        #print('y=%d' % int(y))
        #print('angle=%d' % int(angle))
        #print('mirror=%d' % mirror)
        
        if mirror:
            img = np.flip(img, axis=0)

        rotated_img = imutils.rotate_bound(img, angle)
        rotated_img_cropped = crop_image(rotated_img)
        
        rotated_img_cropped_padded = np.zeros((max_height,max_width))
        
        # compute offset
        x_start_new = (max_width - rotated_img_cropped.shape[1]) // 2
        y_start_new = (max_height - rotated_img_cropped.shape[0]) // 2

        # copy image into center of result image
        rotated_img_cropped_padded[y_start_new:y_start_new+rotated_img_cropped.shape[0],
                            x_start_new:x_start_new+rotated_img_cropped.shape[1]] = rotated_img_cropped
        
        rotated_img_cropped_padded_shifted = ndimage.shift(rotated_img_cropped_padded, (y, x), cval = 0)
        
        return(rotated_img_cropped_padded_shifted)
        

 
    print(all_ref_image_list) 
    print(all_community_image_list)
    print(image_list_cluster_c)
    

    
    image_2d_matrix = get_image_2d_matrix(input_dir)    
  
    xtrans_matrix = alignment_parameters[0] 
    ytrans_matrix = alignment_parameters[1]
    rot_angle_matrix = alignment_parameters[2]
    mirror_matrix = alignment_parameters[3] 
    
    xtrans_matrix_missing = alignment_parameters[4] 
    ytrans_matrix_missing = alignment_parameters[5] 
    rot_angle_matrix_missing = alignment_parameters[6] 
    mirror_matrix_missing = alignment_parameters[7] 

    average_image_wrt_ref_list = []

    for i in range(0,len(all_community_image_list)): #looping through each threshold

        average_image_wrt_ref_list.append([])

        for j in range(0,len(all_community_image_list[i])): #looping through each community

            particle_count_list = []
            for img in all_community_image_list[i][j]:
                particle_count_list.append(particle_count_dict_cluster_c[img])
            particle_count_list = np.array(particle_count_list)
            particle_count_list_norm = particle_count_list/np.sum(particle_count_list)


            curr_community_image_list = all_community_image_list[i][j]
            curr_ref_image_num = all_ref_image_list[i][j]
                
            curr_community_original_image_list = [image_list_cluster_c[x] for x in curr_community_image_list]
            curr_ref_original_image_num = image_list_cluster_c[curr_ref_image_num]

            print(curr_community_image_list)
            print(curr_ref_image_num)
            print(curr_community_original_image_list)
            print(curr_ref_original_image_num)
            
            ###

            max_shape = np.array([0,0])

            for curr_original_image_num in curr_community_original_image_list:
                
                curr_original_image = np.copy(image_2d_matrix[curr_original_image_num,:,:])
                
                if curr_original_image_num > curr_ref_original_image_num:
                    rot_angle = rot_angle_matrix[curr_ref_original_image_num,curr_original_image_num]
                    mirror = mirror_matrix[curr_ref_original_image_num,curr_original_image_num]
                elif curr_original_image_num < curr_ref_original_image_num:
                    rot_angle = rot_angle_matrix_missing[curr_ref_original_image_num,curr_original_image_num]
                    mirror = mirror_matrix_missing[curr_ref_original_image_num,curr_original_image_num]
                else:
                    rot_angle = 0
                    mirror = 0
                    
                if mirror:
                    curr_original_image = np.flip(curr_original_image, axis=0)
                
                rotated_img = imutils.rotate_bound(curr_original_image, rot_angle)
                rotated_img_cropped = crop_image(rotated_img)
                rotated_img_cropped_shape = rotated_img_cropped.shape
                
                if rotated_img_cropped_shape[0] > max_shape[0]:
                    max_shape[0] = rotated_img_cropped_shape[0]
                if rotated_img_cropped_shape[1] > max_shape[1]:
                    max_shape[1] = rotated_img_cropped_shape[1] 
                    
            max_height = max_shape[0]
            max_width = max_shape[1]
            aligned_ref_image_matrix = np.zeros((len(curr_community_original_image_list), max_height, max_width))

            for idx,curr_original_image_num in enumerate(curr_community_original_image_list):
                
                curr_original_image = np.copy(image_2d_matrix[curr_original_image_num,:,:])
                
                if curr_original_image_num > curr_ref_original_image_num:
                    xtrans = xtrans_matrix[curr_ref_original_image_num,curr_original_image_num]
                    ytrans = ytrans_matrix[curr_ref_original_image_num,curr_original_image_num]
                    rot_angle = rot_angle_matrix[curr_ref_original_image_num,curr_original_image_num]
                    mirror = mirror_matrix[curr_ref_original_image_num,curr_original_image_num]
                elif curr_original_image_num < curr_ref_original_image_num:
                    xtrans = xtrans_matrix_missing[curr_ref_original_image_num,curr_original_image_num]
                    ytrans = ytrans_matrix_missing[curr_ref_original_image_num,curr_original_image_num]
                    rot_angle = rot_angle_matrix_missing[curr_ref_original_image_num,curr_original_image_num]
                    mirror = mirror_matrix_missing[curr_ref_original_image_num,curr_original_image_num]
                else:
                    xtrans = 0 
                    ytrans = 0 
                    rot_angle = 0
                    mirror = 0

                print('here')
                print(curr_original_image_num)
                print(xtrans)
                print(ytrans)
                print(rot_angle)
                print(mirror)

                transformed_img = apply_transformations(curr_original_image, xtrans, ytrans, rot_angle, mirror, max_height, max_width)
                
                '''img_scaled = minmax_scale(curr_original_image.ravel(), feature_range=(0,255)).reshape(curr_original_image.shape)
                img_scaled = maintain_aspect_ratio_resize(img_scaled, width=100, height=100)
                avg_img_save_path = '%d_o.png' % curr_original_image_num
                cv2.imwrite(avg_img_save_path, img_scaled)


                img_scaled = minmax_scale(transformed_img.ravel(), feature_range=(0,255)).reshape(transformed_img.shape)
                img_scaled = maintain_aspect_ratio_resize(img_scaled, width=100, height=100)
                avg_img_save_path = '%d_t.png' % curr_original_image_num
                cv2.imwrite(avg_img_save_path, img_scaled)'''

                aligned_ref_image_matrix[idx,:,:] = transformed_img
        
            average_image_wrt_ref = np.zeros((max_height, max_width))
            for k in range(0,aligned_ref_image_matrix.shape[0]): #take weighted average 
                average_image_wrt_ref = average_image_wrt_ref + aligned_ref_image_matrix[k,:,:]*particle_count_list_norm[k]

            #img_scaled = minmax_scale(average_image_wrt_ref.ravel(), feature_range=(0,255)).reshape(average_image_wrt_ref.shape)
            #img_scaled = maintain_aspect_ratio_resize(img_scaled, width=100, height=100)
            #avg_img_save_path = '1_4.png'
            #cv2.imwrite(avg_img_save_path, img_scaled)


        
            average_image_wrt_ref_list[i].append(average_image_wrt_ref)


    return(average_image_wrt_ref_list)
        


def get_community(sparse_matrix, dist_matrix):

    G = nx.from_numpy_matrix(sparse_matrix)
    community = girvan_newman(G)

    node_groups = []
    for com in next(community):
      node_groups.append(list(com))

    node_groups_condensed = []
    for n in node_groups:
        if len(n) > 1:
            node_groups_condensed.append(n)

    node_groups_condensed_flat = [x for sublist in node_groups_condensed for x in sublist]

    cluster_assignment = []
    for node in node_groups_condensed_flat:
        for c_idx,c in enumerate(node_groups_condensed):
            if node in c:
                cluster_assignment.append(c_idx)
  
    if len(node_groups_condensed) > 1:
        sil_input_dist_matrix = (dist_matrix[np.ix_(node_groups_condensed_flat, node_groups_condensed_flat)])
        sil_score = metrics.silhouette_score(sil_input_dist_matrix, cluster_assignment, metric='precomputed')
    else:
        sil_score = -1

    return((node_groups_condensed, sil_score))


def get_dataset_community_dist_metric(community, dist_matrix):

    num_images = dist_matrix.shape[0]
    num_communities = len(community)
    ret = np.zeros((num_images, num_communities))

    for img_i in range(0,num_images):
            
        for c_idx,c in enumerate(community):

            dist_wrt_c = []

            for img_j in c:
                dist_wrt_c.append(dist_matrix[img_i,img_j])

            dist_wrt_c = np.array(dist_wrt_c)
            dist_wrt_c_gt0 = dist_wrt_c[dist_wrt_c > 0]
            ret[img_i,c_idx] = round(np.max(dist_wrt_c_gt0),3)
    
    dataset_community_dist = round(np.mean(np.min(ret, axis=1)),3)
    
    return(dataset_community_dist)


def get_community_probability_membership(community, dist_matrix):

    def assign_probability(dist_vector, exponentiate=True):

        if exponentiate:
            result = np.exp(1./dist_vector)
            result[~np.isfinite(result)] = np.finfo(np.double).max
        else:
            result = 1./dist_vector
            result[~np.isfinite(result)] = np.finfo(np.double).max

        result /= result.sum()    
        return (result)

    num_images = dist_matrix.shape[0]

    img_community_min_dist_map = {}
    img_community_min_prob_map = {}

    for img_i in range(0,num_images):

        img_community_min_dist_map[img_i] = []

        if len(community) > 1:
            
            for c_idx,c in enumerate(community):

                dist_wrt_c = []

                for img_j in c:
                    dist_wrt_c.append(dist_matrix[img_i,img_j])

                dist_wrt_c = np.array(dist_wrt_c)
                dist_wrt_c_gt0 = dist_wrt_c[dist_wrt_c > 0]
                
                img_community_min_dist_map[img_i].append(round(np.min(dist_wrt_c_gt0),3))
        else:

            img_community_min_dist_map[img_i] = [1]


    for img_i in range(0,num_images):
        softmax_prob_min = assign_probability(np.array(img_community_min_dist_map[img_i]), exponentiate=True)
        img_community_min_prob_map[img_i] = softmax_prob_min

    return((img_community_min_dist_map, img_community_min_prob_map))


def get_community_total_counts(img_community_min_prob_map, img_count_map = None):
    
    num_img = len(img_community_min_prob_map.keys())
    num_comm = len(img_community_min_prob_map[0])
    
    img_community_count_matrix = np.zeros((num_img, num_comm))
    
    for img in sorted(img_community_min_prob_map.keys()): 
        if img_count_map is not None:
            img_community_count_matrix[img,:] = img_community_min_prob_map[img]*img_count_map[img]
        else:
            img_community_count_matrix[img,:] = img_community_min_prob_map[img]
            
    return((img_community_count_matrix, np.sum(img_community_count_matrix, axis=0)))


def calc_max_community_weight(community, community_count):

    cluster_weight = community_count/np.sum(np.array(community_count))
    return(np.around(np.max(cluster_weight),2))


def get_cluster_community_map(i, dist_matrix, percentile, particle_count_dict_cluster_c, image_list_cluster_c):

    #print('i start=%d' % i)

    dist_threshold = percentile[i]
    gt_threshold_idx = (np.argwhere(dist_matrix > dist_threshold))
    sparse_matrix = np.copy(dist_matrix)
    sparse_matrix[tuple(gt_threshold_idx.T)] = 0
    community, sil_score = get_community(sparse_matrix, dist_matrix)   

    community_original_images = []
    for outer_val in community:
        tmp = []
        for inner_val in outer_val:
            tmp.append(image_list_cluster_c[inner_val])
        community_original_images.append(tmp)
    
    if len(community) > 0: 
        img_community_min_dist_map, img_community_min_prob_map = get_community_probability_membership(community, dist_matrix)
        img_community_min_prob_matrix, community_prob = get_community_total_counts(img_community_min_prob_map) #not passing in count info
        img_community_count_matrix, community_count = get_community_total_counts(img_community_min_prob_map, particle_count_dict_cluster_c)
        max_community_weight = calc_max_community_weight(community, community_count)
        dataset_community_dist = get_dataset_community_dist_metric(community, dist_matrix)
    else:
        community_count = np.array([])
        max_community_weight = -1
        img_community_min_prob_matrix = np.ones((dist_matrix.shape[0],1))
        dataset_community_dist = 1
        
    ref_image_list = []
    for community_image_list in community:
        ref_img = get_ref_image(dist_matrix, community_image_list, particle_count_dict_cluster_c)
        ref_image_list.append(ref_img)
    
    
    #print('i done=%d' % i)
       
    return(community, community_original_images, dist_threshold, dataset_community_dist, community_count, ref_image_list, max_community_weight, img_community_min_prob_matrix)



def get_cluster_info_parallel(input_dir, corr_cluster_labels, corr_dist_matrix, edge_dist_matrix, corr_only, alignment_parameters, particle_count_dict=None):

    unique_clusters = np.unique(corr_cluster_labels)

    cluster_community_map = {}
    cluster_community_original_image_map = {}
    cluster_dist_threshold_map = {}
    cluster_sil_score_map = {}
    cluster_community_count_map = {}
    cluster_ref_img_map = {}
    cluster_max_community_weight_map = {}
    cluster_img_min_prob_matrix_map = {}
    cluster_dataset_community_dist_map = {}
    cluster_average_image_wrt_ref_list_map = {}
    
    for c in unique_clusters:
        
        image_list_cluster_c = np.genfromtxt('%s/class_average_panel_plots/image_list_cluster_%d.csv' % (input_dir, int(c)), delimiter=',')
        image_list_cluster_c = image_list_cluster_c.astype(int)
         
        particle_count_dict_cluster_c = get_particle_count_dict_cluster(particle_count_dict, image_list_cluster_c)
        edge_dist_matrix_subset = edge_dist_matrix[np.ix_(image_list_cluster_c, image_list_cluster_c)]    
        corr_dist_matrix_subset = corr_dist_matrix[np.ix_(image_list_cluster_c, image_list_cluster_c)]
            
        edge_corr_dist_matrix_subset = edge_dist_matrix_subset/(1-corr_dist_matrix_subset)
        
        edge_corr_percentile = np.percentile(edge_corr_dist_matrix_subset.flatten(), range(1,100))
        edge_corr_percentile = edge_corr_percentile[edge_corr_percentile > 0]
        corr_percentile = np.percentile(corr_dist_matrix_subset.flatten(), range(1,100))
        corr_percentile = corr_percentile[corr_percentile > 0]
        
        max_sil_score = -1
        community_opt = [] 
         
        if corr_only:
            dist_matrix = corr_dist_matrix_subset
            percentile = corr_percentile
        else:
            dist_matrix = edge_corr_dist_matrix_subset
            percentile = edge_corr_percentile

        fd = delayed(get_cluster_community_map)   
        
        out = Parallel(backend="threading", n_jobs=1)(fd(i, dist_matrix, percentile, particle_count_dict_cluster_c, image_list_cluster_c) for i in range(0,np.min([len(percentile),20])))
        print('DONE with cluster=%d' % c)
        community_image_list = [val[0] for val in out] #list of lists of lists (each community is a list of lists)
        community_original_image_list = [val[1] for val in out] #list of lists of lists (each community is a list of lists)
        dist_threshold = [val[2] for val in out]
        dataset_community_dist = [val[3] for val in out]
        community_count = [val[4] for val in out] #list of lists (length of each sublist corresponds to number of community for that threshold)
        ref_image_list = [val[5] for val in out] #list of lists (length of each sublist corresponds to number of community for that threshold)
        max_community_weight = [val[6] for val in out]  #list
        img_community_min_prob_matrix = [val[7] for val in out]
         
        cluster_community_map[c] = community_image_list 
        cluster_community_original_image_map[c] = community_original_image_list
        cluster_dist_threshold_map[c] = dist_threshold
        cluster_dataset_community_dist_map[c] = dataset_community_dist
        cluster_community_count_map[c] = community_count
        cluster_ref_img_map[c] = ref_image_list
        cluster_max_community_weight_map[c] = max_community_weight 
        cluster_img_min_prob_matrix_map[c] = img_community_min_prob_matrix

        print("BEGINNING ref_image_alignment_parameters.get_missing_alignment_parameter")
        rot_angle_matrix_missing, xtrans_matrix_missing, ytrans_matrix_missing, mirror_indicator_matrix_missing = ref_image_alignment_parameters.get_missing_alignment_parameters(image_list_cluster_c, alignment_parameters, community_image_list, ref_image_list, input_dir)
        
        alignment_parameters.append(xtrans_matrix_missing)
        alignment_parameters.append(ytrans_matrix_missing)
        alignment_parameters.append(rot_angle_matrix_missing)
        alignment_parameters.append(mirror_indicator_matrix_missing)

        average_image_wrt_ref_list = get_average_image_wrt_ref(ref_image_list, community_image_list, particle_count_dict_cluster_c, image_list_cluster_c, alignment_parameters, input_dir)
        cluster_average_image_wrt_ref_list_map[c] = average_image_wrt_ref_list

        xtrans_matrix = alignment_parameters[0] 
        ytrans_matrix = alignment_parameters[1]
        rot_angle_matrix = alignment_parameters[2]
        mirror_indicator_matrix = alignment_parameters[3] 



        print(cluster_community_map)
        print(cluster_ref_img_map)

        print(rot_angle_matrix_missing[41,7])
        print(xtrans_matrix_missing[41,7])
        print(ytrans_matrix_missing[41,7])
        print(mirror_indicator_matrix_missing[41,7])

        print(rot_angle_matrix[41,7])
        print(xtrans_matrix[41,7])
        print(ytrans_matrix[41,7])
        print(mirror_indicator_matrix[41,7])

        print(rot_angle_matrix[7,41])
        print(xtrans_matrix[7,41])
        print(ytrans_matrix[7,41])
        print(mirror_indicator_matrix[7,41])



    return((cluster_community_map, cluster_dist_threshold_map, cluster_dataset_community_dist_map, cluster_community_count_map, cluster_ref_img_map, cluster_max_community_weight_map, cluster_img_min_prob_matrix_map, cluster_average_image_wrt_ref_list_map))






def get_optimal_community_map(cluster_dataset_community_dist_map):

    optimal_community_map = {}
    
    for c in cluster_dataset_community_dist_map:
        
        optimal_community_map[c] = np.argmin(cluster_dataset_community_dist_map[c])
    
    return(optimal_community_map)
        



#####
#####




    
def crop_image(img):    
    row_idx, col_idx = np.nonzero(img)
    return(img[np.min(row_idx):np.max(row_idx)+1,np.min(col_idx):np.max(col_idx)+1])


# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)




def save_ref_image(input_dir, cluster_ref_img_map, corr_only):

    image_2d_matrix = get_image_2d_matrix(input_dir)

    for c in cluster_ref_img_map:

        #print('saving ref image for cluster %d' % c)

        image_list_cluster_c = np.genfromtxt('%s/class_average_panel_plots/image_list_cluster_%d.csv' % (input_dir, int(c)), delimiter=',')
        image_list_cluster_c = image_list_cluster_c.astype(int)

        if corr_only:        
            ref_img_save_dir = '%s/histogram_plots/ref_img/corr/cluster_%s' % (input_dir, str(int(c)))
        else:
            ref_img_save_dir = '%s/histogram_plots/ref_img/edge/cluster_%s' % (input_dir, str(int(c)))

        Path(ref_img_save_dir).mkdir(parents=True, exist_ok=True)
        remove_files_in_folder(ref_img_save_dir)
  
        all_ref_img = np.array(cluster_ref_img_map[c])
        all_ref_img_unique = np.sort(np.unique(np.concatenate(all_ref_img)))

        for i in range(0,len(all_ref_img_unique)):
            
            img_num = int(all_ref_img_unique[i])
            img_num_orig = image_list_cluster_c[img_num]
            img = np.copy(np.flip(image_2d_matrix[img_num_orig,:,:], axis=1)) #mrc reads flipped in y
            img = crop_image(img)
            img_scaled = minmax_scale(img.ravel(), feature_range=(0,255)).reshape(img.shape)
            img_scaled = maintain_aspect_ratio_resize(img_scaled, width=100, height=100)
            ref_img_save_path = '%s/%d.png' % (ref_img_save_dir, int(img_num))
            cv2.imwrite(ref_img_save_path, img_scaled)


def save_average_image_wrt_ref(input_dir, cluster_average_image_wrt_ref_list_map, corr_only):


    for c in cluster_average_image_wrt_ref_list_map:

        #print('saving ref image for cluster %d' % c)

        curr_average_image_wrt_ref_list = cluster_average_image_wrt_ref_list_map[c] #list of lists 

        if corr_only:        
            avg_img_save_dir = '%s/histogram_plots/average_image_wrt_ref/corr/cluster_%s' % (input_dir, str(int(c)))
        else:
            avg_img_save_dir = '%s/histogram_plots/average_image_wrt_ref/edge/cluster_%s' % (input_dir, str(int(c)))

        Path(avg_img_save_dir).mkdir(parents=True, exist_ok=True)
        remove_files_in_folder(avg_img_save_dir)
  
        for threshold_idx in range(0,len(curr_average_image_wrt_ref_list)):
            for community_num in range(0,len(curr_average_image_wrt_ref_list[threshold_idx])):
                img_name = '%d_%d' % (threshold_idx, community_num)
                img = curr_average_image_wrt_ref_list[threshold_idx][community_num] 
                img = crop_image(img)
                img_scaled = minmax_scale(img.ravel(), feature_range=(0,255)).reshape(img.shape)
                img_scaled = maintain_aspect_ratio_resize(img_scaled, width=100, height=100)
                avg_img_save_path = '%s/%s.png' % (avg_img_save_dir, img_name)
                cv2.imwrite(avg_img_save_path, img_scaled)

            
def save_img_community_min_prob_matrix(input_dir, cluster_img_min_prob_matrix_map, optimal_community_map, corr_only):

    for c in cluster_img_min_prob_matrix_map:
        
        opt_img_min_prob_matrix = cluster_img_min_prob_matrix_map[c][optimal_community_map[c]]

        if corr_only:        
            prob_save_dir = '%s/histogram_plots/community_probability/corr/cluster_%s' % (input_dir, str(int(c)))
        else:
            prob_save_dir = '%s/histogram_plots/community_probability/edge/cluster_%s' % (input_dir, str(int(c)))

        Path(prob_save_dir).mkdir(parents=True, exist_ok=True)
        remove_files_in_folder(prob_save_dir)
        
        plt.figure(figsize=(16,16))
        sns.heatmap(opt_img_min_prob_matrix, xticklabels=True, yticklabels=True, linewidths=.5)
        plt.savefig('%s/prob_heatmap.png' % prob_save_dir)
        
        

def hist_wrapper(input_dir):

    corr_cluster_labels = np.genfromtxt('%s/spectral_clustering/corr_cluster_labels.csv' %  input_dir, delimiter=',') 
    corr_dist_matrix = np.genfromtxt('%s/pairwise_matrix/corr_dist_matrix.csv' % input_dir, delimiter=',')
    edge_dist_matrix = np.genfromtxt('%s/pairwise_matrix/edge_dist_matrix.csv' % input_dir, delimiter=',')

    xtrans_matrix = np.genfromtxt('%s/pairwise_matrix/xtrans_matrix.csv' % input_dir, delimiter=',')
    ytrans_matrix = np.genfromtxt('%s/pairwise_matrix/ytrans_matrix.csv' % input_dir, delimiter=',')
    rot_matrix = np.genfromtxt('%s/pairwise_matrix/rot_matrix.csv' % input_dir, delimiter=',')
    mirror_indicator_matrix = np.genfromtxt('%s/pairwise_matrix/mirror_indicator_matrix.csv' % input_dir, delimiter=',')

    alignment_parameters = [xtrans_matrix, ytrans_matrix, rot_matrix, mirror_indicator_matrix]    

    particle_count_dict = get_particle_count(input_dir)

    hist_data = {}

    save_path = '%s/histogram_plots/raw_data' % input_dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    for i in range(0,2):

        if i == 0:
            corr_only = False
            dist_metric_str = 'edge'
            print('edge based community detection')
        else:
            corr_only = True
            dist_metric_str = 'corr'
            print('correlation based community detection')
         
        cluster_community_map, cluster_dist_threshold_map, cluster_dataset_community_dist_map, cluster_community_count_map, cluster_ref_img_map, cluster_max_community_weight_map, cluster_img_min_prob_matrix_map, cluster_average_image_wrt_ref_list_map = get_cluster_info_parallel(input_dir, corr_cluster_labels, corr_dist_matrix, edge_dist_matrix, corr_only, alignment_parameters, particle_count_dict)

        optimal_community_map = get_optimal_community_map(cluster_dataset_community_dist_map)

        hist_data[dist_metric_str] = [cluster_community_map, cluster_dist_threshold_map, cluster_dataset_community_dist_map, cluster_community_count_map, cluster_ref_img_map, cluster_max_community_weight_map, optimal_community_map]
        print('saving image probility matrix')
        save_img_community_min_prob_matrix(input_dir, cluster_img_min_prob_matrix_map, optimal_community_map, corr_only)        
        print('saving ref images')
        save_ref_image(input_dir, cluster_ref_img_map, corr_only)
        print('saving avg images wrt ref')
        save_average_image_wrt_ref(input_dir, cluster_average_image_wrt_ref_list_map, corr_only)


    print('saving %s' % save_path)
    save_obj(hist_data, '%s/hist_data' % save_path)
        

     







