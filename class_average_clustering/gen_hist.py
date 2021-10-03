import sys
import os 
import argparse 
import numpy as np
import mrcfile
from joblib import Parallel, effective_n_jobs, delayed
from sklearn import cluster
from sklearn import metrics
import math
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
from sklearn.preprocessing import minmax_scale
from pathlib import Path
from cv2 import *
#import pandas as pd
from helper_functions import save_obj, get_image_2d_matrix, get_particle_count, get_particle_count_dict_cluster, remove_files_in_folder 

os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib.pyplot as plt
import seaborn as sns


def get_ref_image(dist_matrix, image_list, particle_count):
     
    #get image with closest median distance to all other images 
    median_dist_list = []
    particle_count_list = []
    for i in image_list:
        dist_wrt_i = []
        particle_count_list.append(particle_count[i])
        for j in image_list:
            dist_ij = dist_matrix[i,j]
            dist_wrt_i.append(dist_ij)
        dist_wrt_i = np.array(dist_wrt_i)
        median_dist_list.append(np.median(dist_wrt_i))
    
    if len(image_list) > 2:
        median_dist_list = np.array(median_dist_list)
        ref_img = image_list[np.argmin(median_dist_list)]
    else:
        ref_img = image_list[np.argmax(np.array(particle_count_list))]
    
    return(ref_img)


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


def get_min_community_threshold(th):

    if th >= 20:
        min_community_threshold = 3
    else:
        min_community_threshold = 2
    
    return(min_community_threshold)


#obsolete 
def get_median_dist_between_clusterij(dist_matrix, image_list_i, image_list_j):
    
    #get image with closest median distance to all other images 
    dist_list = []
    for i in image_list_i:
        for j in image_list_j:
            dist_ij = dist_matrix[i,j]
            dist_list.append(dist_ij)
        
    return(np.median(np.array(dist_list)))


def calc_max_community_weight(community, community_count):

    cluster_weight = community_count/np.sum(np.array(community_count))
    return(np.around(np.max(cluster_weight),2))


#obsolete 
def calc_normalized_community_dist_metric(cluster_community_map, cluster_community_count_map, cluster_num, dist_matrix):

    if len(community) == 1:
        return(0)
 
    cluster_ij_median_dist_list = []
    min_cluster_weight_ij_list = []
        
    ##calculate between community dist
    for i in range(0,(len(cluster_community_map[cluster_num])-1)):

        for j in range(i+1,len(cluster_community_map[cluster_num])):

            comm_i = cluster_community_map[cluster_num][i]
            comm_j = cluster_community_map[cluster_num][j]
            
            cluster_ij_median_dist = get_median_dist_between_clusterij(dist_matrix, comm_i, comm_j)
            min_cluster_weight_ij = np.min([cluster_weight[i], cluster_weight[j]])

            cluster_ij_median_dist_list.append(cluster_ij_median_dist)
            min_cluster_weight_ij_list.append(min_cluster_weight_ij)
            
    cluster_ij_median_dist_list = np.array(cluster_ij_median_dist_list)
    min_cluster_weight_ij_list = np.array(min_cluster_weight_ij_list)
 
    between_community_variance_numerator = np.dot(cluster_ij_median_dist_list, min_cluster_weight_ij_list)
    between_community_variance_denominator = np.sum(min_cluster_weight_ij_list)
     
    ##split max cluster
    cluster_weight_argsorted = np.argsort(cluster_weight)[::-1]
    largest_cluster_idx = cluster_weight_argsorted[0] 
    largest_cluster_weight =  cluster_weight[largest_cluster_idx]
    largest_cluster_split_weight = []
    
    tmp = largest_cluster_weight
    split_weight = 1-largest_cluster_weight
    while tmp > 0:
        largest_cluster_split_weight.append(np.min([tmp, split_weight]))
        tmp = tmp - split_weight
    
    pairwise_max_community_weight_list = np.repeat(np.max(np.array(largest_cluster_split_weight)),len(largest_cluster_split_weight)*(len(largest_cluster_split_weight)-1)/2)
 
    max_cluster_split_numerator = np.sum(pairwise_max_community_weight_list) 
    max_cluster_split_denominator = max_cluster_split_numerator

    normalized_between_community_dist_metric = 1-((between_community_variance_numerator + max_cluster_split_numerator)/(between_community_variance_denominator + max_cluster_split_denominator))
    normalized_between_community_dist_metric = np.round(normalized_between_community_dist_metric,2)
     
    return(normalized_between_community_dist_metric)

#obsolete
def calc_normalized_community_dist_metric_parallel(community, community_count, dist_matrix):

    if len(community) == 1:
        return(0)
    
    cluster_weight = community_count/np.sum(np.array(community_count))
     
    cluster_ij_median_dist_list = []
    min_cluster_weight_ij_list = []
        
    ##calculate between community dist
    for i in range(0,(len(community)-1)):

        for j in range(i+1,len(community)):

            comm_i = community[i]
            comm_j = community[j]
            
            cluster_ij_median_dist = get_median_dist_between_clusterij(dist_matrix, comm_i, comm_j)
            min_cluster_weight_ij = np.min([cluster_weight[i], cluster_weight[j]])

            cluster_ij_median_dist_list.append(cluster_ij_median_dist)
            min_cluster_weight_ij_list.append(min_cluster_weight_ij)
            
    cluster_ij_median_dist_list = np.array(cluster_ij_median_dist_list)
    min_cluster_weight_ij_list = np.array(min_cluster_weight_ij_list)
 
    between_community_variance_numerator = np.dot(cluster_ij_median_dist_list, min_cluster_weight_ij_list)
    between_community_variance_denominator = np.sum(min_cluster_weight_ij_list)
     
    ##split max cluster
    cluster_weight_argsorted = np.argsort(cluster_weight)[::-1]
    largest_cluster_idx = cluster_weight_argsorted[0] 
    largest_cluster_weight =  cluster_weight[largest_cluster_idx]
    largest_cluster_split_weight = []
    
    tmp = largest_cluster_weight
    if tmp > .99:
        return(0) #to avoid too many splits
    split_weight = 1-largest_cluster_weight
    while tmp > 0:
        largest_cluster_split_weight.append(np.min([tmp, split_weight]))
        tmp = tmp - split_weight
    
    pairwise_max_community_weight_list = np.repeat(np.max(np.array(largest_cluster_split_weight)),len(largest_cluster_split_weight)*(len(largest_cluster_split_weight)-1)/2)
 
    max_cluster_split_numerator = np.sum(pairwise_max_community_weight_list) 
    max_cluster_split_denominator = max_cluster_split_numerator
    
    normalized_between_community_dist_metric = 1-((between_community_variance_numerator + max_cluster_split_numerator)/(between_community_variance_denominator + max_cluster_split_denominator))
    normalized_between_community_dist_metric = np.round(normalized_between_community_dist_metric,2)
    
    return(normalized_between_community_dist_metric)




def get_cluster_community_map(i, dist_matrix, percentile, particle_count_dict_cluster_c, image_list_hclust_cluster_c):

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
            tmp.append(image_list_hclust_cluster_c[inner_val])
        community_original_images.append(tmp)
    
    if len(community) > 0: 
        img_community_min_dist_map, img_community_min_prob_map = get_community_probability_membership(community, dist_matrix)
        img_community_min_prob_matrix, community_prob =  get_community_total_counts(img_community_min_prob_map) #not passing in count info
        img_community_count_matrix, community_count = get_community_total_counts(img_community_min_prob_map, particle_count_dict_cluster_c)
        max_community_weight = calc_max_community_weight(community, community_count)
        dataset_community_dist = get_dataset_community_dist_metric(community, dist_matrix)
    else:
        community_count = np.array([])
        max_community_weight = -1
        img_community_min_prob_matrix = np.ones((dist_matrix.shape[0],1))
        dataset_community_dist = 1
        
    ref_image_list = []
    for image_list in community:
        ref_img = get_ref_image(dist_matrix, image_list, particle_count_dict_cluster_c)
        ref_image_list.append(ref_img)
    
    
    #print('i done=%d' % i)
       
    return(community, community_original_images, dist_threshold, dataset_community_dist, community_count, ref_image_list, max_community_weight, img_community_min_prob_matrix)



def get_cluster_info_parallel(input_dir, corr_cluster_labels, corr_dist_matrix, edge_dist_matrix, corr_only, particle_count_dict=None):

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
    
    for c in unique_clusters:
        
        image_list_hclust_cluster_c = np.genfromtxt('%s/class_average_panel_plots/image_list_hclust_cluster_%d.csv' % (input_dir, int(c)), delimiter=',')
        image_list_hclust_cluster_c = image_list_hclust_cluster_c.astype(int)
        
        min_community_threshold = get_min_community_threshold(len(image_list_hclust_cluster_c)) 
        
        particle_count_dict_cluster_c = get_particle_count_dict_cluster(particle_count_dict, image_list_hclust_cluster_c)
        edge_dist_matrix_subset = edge_dist_matrix[np.ix_(image_list_hclust_cluster_c, image_list_hclust_cluster_c)]    
        corr_dist_matrix_subset = corr_dist_matrix[np.ix_(image_list_hclust_cluster_c, image_list_hclust_cluster_c)]
            
        edge_corr_dist_matrix_subset = edge_dist_matrix_subset/(1-corr_dist_matrix_subset)
        #normalized between 0-1 so we can assign a probability in the same manner as we did for correlation based distance
        #edge_corr_dist_matrix_subset_norm = (edge_corr_dist_matrix_subset - np.min(edge_corr_dist_matrix_subset.flatten())) / (np.max(edge_corr_dist_matrix_subset.flatten()) - np.min(edge_corr_dist_matrix_subset.flatten())) 

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
        out = Parallel(backend="threading", n_jobs=1)(fd(i, dist_matrix, percentile, particle_count_dict_cluster_c, image_list_hclust_cluster_c) for i in range(0,np.min([len(percentile),20])))
        print('DONE with cluster=%d' % c)
        community = [val[0] for val in out] #list of lists of lists (each community is a list of lists)
        community_original_images = [val[1] for val in out] #list of lists of lists (each community is a list of lists)
        dist_threshold = [val[2] for val in out]
        dataset_community_dist = [val[3] for val in out]
        community_count = [val[4] for val in out] #list of lists (length of each sublist corresponds to number of community for that threshold)
        ref_image_list = [val[5] for val in out] #list of lists (length of each sublist corresponds to number of community for that threshold)
        max_community_weight = [val[6] for val in out]  #list
        img_community_min_prob_matrix = [val[7] for val in out]
         
        cluster_community_map[c] = community          
        cluster_community_original_image_map[c] = community_original_images
        cluster_dist_threshold_map[c] = dist_threshold
        cluster_dataset_community_dist_map[c] = dataset_community_dist
        cluster_community_count_map[c] = community_count
        cluster_ref_img_map[c] = ref_image_list
        cluster_max_community_weight_map[c] = max_community_weight 
        cluster_img_min_prob_matrix_map[c] = img_community_min_prob_matrix
        
    return((cluster_community_map, cluster_dist_threshold_map, cluster_dataset_community_dist_map, cluster_community_count_map, cluster_ref_img_map, cluster_max_community_weight_map, cluster_img_min_prob_matrix_map))




#obsolete
def get_optimal_community_map_old(input_dir, cluster_community_map, cluster_sil_score_map):

    optimal_community_map = {}

    for c in sorted(cluster_sil_score_map.keys()):

        max_sil_score = -1
        max_sil_score_idx = -1
        community_cluster_c = cluster_community_map[c]
        sil_score_cluster_c = cluster_sil_score_map[c]

        image_list_hclust_cluster_c = np.genfromtxt('%s/class_average_panel_plots/image_list_hclust_cluster_%d.csv' % (input_dir, int(c)), delimiter=',')
        min_community_threshold = get_min_community_threshold(len(image_list_hclust_cluster_c))

        for i in range(0,len(sil_score_cluster_c)):
            
            curr_sil_score = sil_score_cluster_c[i]
            curr_community = community_cluster_c[i]

            if ((curr_sil_score > max_sil_score) and (len(curr_community) >= min_community_threshold)):       
                max_sil_score = curr_sil_score 
                max_sil_score_idx = i

        if max_sil_score == -1: #assign first cluster as optimal cluster 

            for i in range(0,len(sil_score_cluster_c)):
                
                curr_sil_score = sil_score_cluster_c[i]
                curr_community = community_cluster_c[i]

                if len(curr_community) > 0: 
                    max_sil_score = -1 
                    max_sil_score_idx = i  
        
        optimal_community_map[c] = max_sil_score_idx
    
    return(optimal_community_map)


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

        image_list_hclust_cluster_c = np.genfromtxt('%s/class_average_panel_plots/image_list_hclust_cluster_%d.csv' % (input_dir, int(c)), delimiter=',')
        image_list_hclust_cluster_c = image_list_hclust_cluster_c.astype(int)

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
            img_num_orig = image_list_hclust_cluster_c[img_num]
            img = np.copy(np.flip(image_2d_matrix[img_num_orig,:,:], axis=1)) #mrc reads flipped in y
            img = crop_image(img)
            img_scaled = minmax_scale(img.ravel(), feature_range=(0,255)).reshape(img.shape)
            img_scaled = maintain_aspect_ratio_resize(img_scaled, width=100, height=100)
            ref_img_save_path = '%s/%d.png' % (ref_img_save_dir, int(img_num))
            cv2.imwrite(ref_img_save_path, img_scaled)
            

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
        
        

'''def save_hist(protein, cluster_num, cluster_community_map, cluster_community_count_map, cluster_ref_img_map, cluster_ref_img_locs, cluster_max_community_weight_map, corr_only):

    if corr_only:
        hist_save_path = './output/%s/preferred_orientation_hist/corr_only' % protein
    else:
        hist_save_path = './output/%s/preferred_orientation_hist/edge_corr_ratio' % protein

    print('saving histogram in %s' % hist_save_path)
 
    Path(hist_save_path).mkdir(parents=True, exist_ok=True)

    y=cluster_community_count_map[cluster_num]/(np.sum(cluster_community_count_map[cluster_num]))
    x=np.linspace(1,len(y),len(y))

    print(cluster_ref_img_map)
    ref_imgs_cluster_c = [x[0] for x in cluster_ref_img_map[cluster_num]]

    all_imgs_cluster_c = []
    for community_img in cluster_community_map[cluster_num]:
        all_imgs_cluster_c.append("-".join(str(x) for x in community_img))
        
    plot_df = pd.DataFrame({'x': x, 'y': y, 'ref_img': ref_imgs_cluster_c, 'all_img': all_imgs_cluster_c})

    ref_img_list = cluster_ref_img_locs[cluster_num]
    print(ref_img_list)
    
    plot_title = 'Normalized Community Distance = %s' % cluster_max_community_weight_map[cluster_num]    

    fig = px.bar(plot_df, x='x', y='y',
                hover_data=['ref_img', 'all_img'])

    fig.update_layout(yaxis_range=[0,np.max(y)*2], title = plot_title,
                      yaxis_title = 'Percentage of Particles in Cluster %d' % cluster_num, xaxis_title = 'Community Number')
    # add images
    for i,src,yy in zip(range(0,len(ref_img_list)),ref_img_list,y):
        logo = base64.b64encode(open(src, 'rb').read())
        fig.add_layout_image(
            source='data:image/png;base64,{}'.format(logo.decode()),
            xref="x",
            yref="y",
            x=x[i],
            y=yy+.02,
            xanchor="center",
            yanchor="bottom",
            sizex=.5,
            sizey=.5,
        )
    #fig.show() 

    fig.write_html('%s/cluster_%s.html' % (hist_save_path, str(int(cluster_num))))'''


'''for protein in ['cryosparc_P50_J436_020_class_averages', 'hLonOpenClosed_class_averages', 'KaiC_class_averages', 'cryosparc_P127_J4_020_class_averages']:

    print(protein)

    corr_cluster_labels = np.genfromtxt('./output/%s/clean/spectral/corr_cluster_labels.csv' %  protein, delimiter=',') 
    corr_dist_matrix = np.genfromtxt('./output/%s/clean/corr_dist_matrix.csv' % protein, delimiter=',')
    edge_dist_matrix = np.genfromtxt('./output/%s/clean/edge_dist_matrix.csv' % protein, delimiter=',')

    particle_count_clean_path = './raw_data/%s_particle_counts_clean' % protein
    if Path('%s.pkl' % particle_count_clean_path).exists():
        particle_count_dict = load_obj(particle_count_clean_path)
    else:
        particle_count_dict = load_obj('./raw_data/%s_particle_counts' % protein)

    hist_data = {}

    save_path = './output/%s/histogram_plots/raw_data' % protein
    Path(save_path).mkdir(parents=True, exist_ok=True)

    for i in range(0,2):

        print('i=%d' % i)

        if i == 0:
            corr_only = False
            dist_metric_str = 'edge'
            print('edge based community detection')
        else:
            corr_only = True
            dist_metric_str = 'corr'
            print('correlation based community detection')
         
        cluster_community_map, cluster_dist_threshold_map, cluster_sil_score_map, cluster_community_count_map, cluster_ref_img_map, cluster_max_community_weight_map = get_cluster_info_parallel(protein, corr_cluster_labels, corr_dist_matrix, edge_dist_matrix, corr_only, particle_count_dict)

        #print(cluster_community_map)
        #print(cluster_dist_threshold_map)
        #print(cluster_sil_score_map)
        #print(cluster_community_count_map)
        #print(cluster_ref_img_map)
        #print(cluster_max_community_weight_map)

        optimal_community_map = get_optimal_community_map(protein, cluster_sil_score_map)

        hist_data[dist_metric_str] = [cluster_community_map, cluster_dist_threshold_map, cluster_sil_score_map, cluster_community_count_map, cluster_ref_img_map, cluster_max_community_weight_map, optimal_community_map]
        
        save_ref_image(protein, cluster_ref_img_map, corr_only)

    print(hist_data)
    print('saving %s' % save_path)
    save_obj(hist_data, '%s/hist_data' % save_path)'''
     
def hist_wrapper(input_dir):

    corr_cluster_labels = np.genfromtxt('%s/spectral_clustering/corr_cluster_labels.csv' %  input_dir, delimiter=',') 
    corr_dist_matrix = np.genfromtxt('%s/pairwise_matrix/corr_dist_matrix.csv' % input_dir, delimiter=',')
    edge_dist_matrix = np.genfromtxt('%s/pairwise_matrix/edge_dist_matrix.csv' % input_dir, delimiter=',')

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
         
        cluster_community_map, cluster_dist_threshold_map, cluster_dataset_community_dist_map, cluster_community_count_map, cluster_ref_img_map, cluster_max_community_weight_map, cluster_img_min_prob_matrix_map = get_cluster_info_parallel(input_dir, corr_cluster_labels, corr_dist_matrix, edge_dist_matrix, corr_only, particle_count_dict)

        optimal_community_map = get_optimal_community_map(cluster_dataset_community_dist_map)

        hist_data[dist_metric_str] = [cluster_community_map, cluster_dist_threshold_map, cluster_dataset_community_dist_map, cluster_community_count_map, cluster_ref_img_map, cluster_max_community_weight_map, optimal_community_map]
        print('saving image probility matrix')
        save_img_community_min_prob_matrix(input_dir, cluster_img_min_prob_matrix_map, optimal_community_map, corr_only)        
        print('saving ref images')
        save_ref_image(input_dir, cluster_ref_img_map, corr_only)

    print('saving %s' % save_path)
    save_obj(hist_data, '%s/hist_data' % save_path)
        

'''if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Directory to where pairwise matrices were saved (default directory will be called xyz_summary, where xyz refers to name of original mrc file)", required=True)
    args = parser.parse_args()

    filepath_txt_file = '%s/filepath.txt' % args.input_dir 
    if Path(filepath_txt_file).exists() == False:
        sys.exit('input_dir is invalid - must be the directory where pairwise matrices were saved')

    corr_cluster_labels = np.genfromtxt('%s/spectral_clustering/corr_cluster_labels.csv' %  args.input_dir, delimiter=',') 
    corr_dist_matrix = np.genfromtxt('%s/pairwise_matrix/corr_dist_matrix.csv' % args.input_dir, delimiter=',')
    edge_dist_matrix = np.genfromtxt('%s/pairwise_matrix/edge_dist_matrix.csv' % args.input_dir, delimiter=',')

    particle_count_dict = get_particle_count(args.input_dir)

    hist_data = {}

    save_path = '%s/histogram_plots/raw_data' % args.input_dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    for i in range(0,2):

        print('i=%d' % i)

        if i == 0:
            corr_only = False
            dist_metric_str = 'edge'
            print('edge based community detection')
        else:
            corr_only = True
            dist_metric_str = 'corr'
            print('correlation based community detection')
         
        cluster_community_map, cluster_dist_threshold_map, cluster_sil_score_map, cluster_community_count_map, cluster_ref_img_map, cluster_max_community_weight_map, cluster_img_min_prob_matrix_map = get_cluster_info_parallel(args.input_dir, corr_cluster_labels, corr_dist_matrix, edge_dist_matrix, corr_only, particle_count_dict)

        optimal_community_map = get_optimal_community_map(args.input_dir, cluster_community_map, cluster_sil_score_map)

        hist_data[dist_metric_str] = [cluster_community_map, cluster_dist_threshold_map, cluster_sil_score_map, cluster_community_count_map, cluster_ref_img_map, cluster_max_community_weight_map, optimal_community_map]
        print('saving image probility matrix')
        save_img_community_min_prob_matrix(args.input_dir, cluster_img_min_prob_matrix_map, optimal_community_map, corr_only)        
        print('saving ref images')
        save_ref_image(args.input_dir, cluster_ref_img_map, corr_only)

    print('saving %s' % save_path)
    save_obj(hist_data, '%s/hist_data' % save_path)'''
     







