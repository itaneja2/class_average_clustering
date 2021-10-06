import sys
import argparse
import math
import csv
import shutil 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mrcfile
import collections 
from cv2 import *
from sklearn.preprocessing import minmax_scale
import pandas as pd
from pathlib import Path
from sklearn import cluster
from sklearn import metrics
import scipy.signal
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn import cluster
from sklearn import metrics
from helper_functions import load_obj, get_image_2d_matrix, get_particle_count, get_particle_count_dict_cluster, remove_files_in_folder 
from gen_hist_v2 import hist_wrapper 

def calc_spectral_cluster_labels(input_dir, dist_type, num_clusters):

    dist_matrix = np.genfromtxt('%s/pairwise_matrix/%s_dist_matrix.csv' % (input_dir, dist_type), delimiter=',') 
    
    if dist_type == 'corr':
        spectral_dist_matrix = 1-dist_matrix
    else:
        spectral_dist_matrix = dist_matrix
        spectral_dist_matrix = np.exp(-spectral_dist_matrix ** 2 / (2. * delta ** 2))

    spectral_model_dict = {}
    spectral_model_sil_score = {}    
    
    optimal_cluster = ''
    max_sil_score = -1
    eps = .06

    if num_clusters is not None:

        clustering = cluster.SpectralClustering(n_clusters=num_clusters, n_init = 100, affinity='precomputed', assign_labels='kmeans').fit(spectral_dist_matrix)
        
        if num_clusters > 1:
            sil_score = metrics.silhouette_score(dist_matrix, clustering.labels_, metric='precomputed')
        else:
            sil_score = -1

        spectral_model_sil_score[num_clusters] = sil_score
        optimal_cluster = clustering
        max_sil_score = sil_score

        print("Num Clusters = 1")
        print(sil_score)
        print(clustering.labels_)

    else:
  
        for num_clusters in range(2,5):
            
            clustering = cluster.SpectralClustering(n_clusters=num_clusters, n_init = 100, affinity='precomputed', assign_labels='kmeans').fit(spectral_dist_matrix)
            sil_score = metrics.silhouette_score(dist_matrix, clustering.labels_, metric='precomputed')
            spectral_model_sil_score[num_clusters] = sil_score
            
            if max_sil_score == -1:
                max_sil_score = sil_score
                optimal_cluster = clustering
            elif sil_score > (max_sil_score-eps):
                if sil_score > .3: #if selecting cluster with lower sil score, it should still be a relatively 'good' cluster 
                    max_sil_score = sil_score
                    optimal_cluster = clustering

            print("Num Clusters = %d" % num_clusters)
            print(sil_score)
            print(clustering.labels_)

    cluster_output_folder = '%s/spectral_clustering' % input_dir
    Path(cluster_output_folder).mkdir(parents=True, exist_ok=True)
    remove_files_in_folder(cluster_output_folder) 
    np.savetxt('%s/%s_cluster_labels.csv' % (cluster_output_folder, dist_type), optimal_cluster.labels_) 
    np.savetxt('%s/%s_silhouette_score.csv' % (cluster_output_folder, dist_type), np.array([max_sil_score]))



def plot_dist_matrix(input_dir, dist_type, clustering_method):

    clustering_dir = '%s/%s' % (input_dir,clustering_method)
    dist_matrix = np.genfromtxt('%s/pairwise_matrix/%s_dist_matrix.csv' % (input_dir, dist_type), delimiter=',') 
    cluster_labels = np.genfromtxt('%s/%s_cluster_labels.csv' % (clustering_dir, dist_type), delimiter=',')

    output_dir = '%s/plots' % clustering_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    counts = collections.Counter(cluster_labels)
    cluster_labels_sorted_by_freq = sorted(cluster_labels, key=lambda x: -counts[x])

    cluster_labels_sorted_by_freq_unique = []
    for val in cluster_labels_sorted_by_freq:
        if val != -1 and val not in cluster_labels_sorted_by_freq_unique:
            cluster_labels_sorted_by_freq_unique.append(val)
    
    if -1 in cluster_labels_sorted_by_freq:
        cluster_labels_sorted_by_freq_unique.append(-1)

    permuted_indices = []
    for val in cluster_labels_sorted_by_freq_unique:
        permuted_indices.append(list(np.where(cluster_labels==val)[0]))

    permuted_indices = [item for sublist in permuted_indices for item in sublist]

    cluster_labels_permuted = cluster_labels[permuted_indices]
    dist_matrix_permuted = dist_matrix[:,permuted_indices]
    dist_matrix_permuted = dist_matrix_permuted[permuted_indices,:]

    image_labels = []
    for i in range(0,len(cluster_labels_permuted)):
        image_labels.append(str(int(permuted_indices[i])) + '_' + str(int(cluster_labels_permuted[i])))

    dist_matrix_permuted_df = pd.DataFrame(dist_matrix_permuted, columns = image_labels, index = image_labels)

    plt.figure(figsize=(16,16))
    sns.heatmap(dist_matrix_permuted_df, linewidths=.5, xticklabels=True, yticklabels=True, annot_kws={"size": 35 / np.sqrt(len(dist_matrix_permuted_df))})
    plt.savefig('%s/%s_heatmap.png' % (output_dir, dist_type))


def plot_hierarchical_cluster(input_dir, dist_type, clustering_method):

    clustering_dir = '%s/%s' % (input_dir,clustering_method)
    dist_matrix = np.genfromtxt('%s/pairwise_matrix/%s_dist_matrix.csv' % (input_dir, dist_type), delimiter=',') 

    output_dir = '%s/plots' % clustering_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16,16))
    sns.clustermap(dist_matrix, xticklabels=True, yticklabels=True, annot_kws={"size": 35 / np.sqrt(len(dist_matrix))})
    plt.savefig('%s/%s_heatmap.png' % (output_dir, dist_type))

    
def gen_image_panel(input_dir, image_2d_matrix, particle_count_dict):

    output_dir = '%s/%s' % (input_dir,'class_average_panel_plots')
    corr_dist_matrix = np.genfromtxt('%s/pairwise_matrix/corr_dist_matrix.csv' % input_dir, delimiter=',') 
    edge_dist_matrix = np.genfromtxt('%s/pairwise_matrix/edge_dist_matrix.csv' % input_dir, delimiter=',') 
 
    cluster_labels = np.genfromtxt('%s/spectral_clustering/corr_cluster_labels.csv' % input_dir, delimiter=',')

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    remove_files_in_folder(output_dir)

    cluster_mapping = {}
    for i in range(0,len(cluster_labels)):
        if cluster_labels[i] not in cluster_mapping:
            cluster_mapping[cluster_labels[i]] = [i]
        else:
            cluster_mapping[cluster_labels[i]].append(i)

    for c in sorted(cluster_mapping.keys()):

        image_list = np.array(cluster_mapping[c]) 
        edge_dist_matrix_subset = edge_dist_matrix[np.ix_(image_list, image_list)]
        corr_dist_matrix_subset = 1-corr_dist_matrix[np.ix_(image_list, image_list)]
        edge_corr_ratio_dist_matrix_subset = edge_dist_matrix_subset/corr_dist_matrix_subset
        #edge_corr_ratio_dist_matrix_subset = edge_dist_matrix_subset/np.square(corr_dist_matrix_subset)


             
        corresponding_images, cluster_sub_labels = calc_hierarchical_cluster(edge_corr_ratio_dist_matrix_subset)
        image_list_hclust_ordered = image_list[corresponding_images]

        particle_count_dict_cluster_c = get_particle_count_dict_cluster(particle_count_dict, image_list_hclust_ordered)

        '''counts = collections.Counter(cluster_sub_labels)
        cluster_sub_labels_sorted_by_freq = sorted(cluster_sub_labels, key=lambda x: -counts[x])

        cluster_sub_labels_sorted_by_freq_unique = []
        for val in cluster_sub_labels_sorted_by_freq:
            if val != -1 and val not in cluster_sub_labels_sorted_by_freq_unique:
                cluster_sub_labels_sorted_by_freq_unique.append(val)

        image_list_sorted = []
        for s in cluster_sub_labels_sorted_by_freq_unique:
            image_list_sorted.append(image_list_hclust_ordered[np.where(cluster_sub_labels==s)[0]])
        image_list_sorted = [item for sublist in image_list_sorted for item in sublist]'''
        
        gen_mrc_file(int(c), image_list_hclust_ordered, image_2d_matrix, output_dir)
        gen_png_panel(int(c), image_list_hclust_ordered, image_2d_matrix, output_dir)
            
        filename = '%s/particle_count_cluster_%s.csv' % (output_dir, str(int(c)))
        with open(filename,'w') as f:
            writer = csv.writer(f)
            for key in particle_count_dict_cluster_c.keys():
                writer.writerow([key, particle_count_dict_cluster_c[key]])

        filename = '%s/dist_matrix_cluster_%s.csv' % (output_dir, str(int(c)))
        np.savetxt(filename, edge_corr_ratio_dist_matrix_subset, delimiter=',')
        filename = '%s/image_list_cluster_%s.csv' % (output_dir, str(int(c)))
        np.savetxt(filename, image_list_hclust_ordered, delimiter=',')



def calc_hierarchical_cluster(dist_matrix):   

    dendogram_nodes = [str(i) for i in range(1,dist_matrix.shape[0]+1)]

    clusters = {}
    linkage_mat = linkage(squareform(dist_matrix), method = 'average', optimal_ordering=True)
    plt.figure(figsize=(150,150))
    dn = dendrogram(linkage_mat)
    
    #print('saving dendogram')
    #plt.savefig('%s/denodgram.png' % dendogram_save_path, format='png', bbox_inches='tight')
    #print('dendogram saved')

    cluster_labels = ['none'] * dist_matrix.shape[0]
    for xs, c in zip(dn['icoord'], dn['color_list']):
        for xi in xs:
            if xi % 10 == 5:
                cluster_labels[(int(xi)-5) // 10] = c

    cluster_labels = np.array(cluster_labels)    
    tree_leaves = leaves_list(linkage_mat)

    return((tree_leaves, cluster_labels))


def gen_mrc_file(cluster_num, image_list, image_2d_matrix, save_path):

    filename = '%s/cluster_%s.mrc' % (save_path, cluster_num)
    print('saving %s' % filename)
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(image_2d_matrix[image_list,:,:]) 


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



def gen_png_panel(cluster_num, image_list, image_2d_matrix, save_path):

    filename = '%s/cluster_%s.png' % (save_path, cluster_num)
    print('saving %s' % filename)

    num_images = len(image_list)
    nearest_sqrt = math.ceil(math.sqrt(num_images))

    num_rows = nearest_sqrt
    if nearest_sqrt*nearest_sqrt > num_images:
        num_rows = nearest_sqrt-1
        if num_rows*nearest_sqrt < num_images:
            num_rows = nearest_sqrt
    
    num_cols = nearest_sqrt        

    IMG_WIDTH = 100
    IMG_HEIGHT = 100 

    
    tiled_image = np.zeros((num_rows*IMG_HEIGHT,num_cols*IMG_WIDTH))
    max_dim = np.max([num_cols,num_rows])

    for i in range(0,image_2d_matrix.shape[0]):

        img = image_2d_matrix[image_list[i],:,:]
        img_scaled = minmax_scale(img.ravel(), feature_range=(0,255)).reshape(img.shape)
        img_scaled = maintain_aspect_ratio_resize(img_scaled, width=IMG_WIDTH, height=IMG_HEIGHT)
        
        col_idx = i % max_dim
        row_idx = int(i / max_dim)
        
        tiled_image[row_idx*IMG_WIDTH:(row_idx+1)*IMG_WIDTH,col_idx*IMG_WIDTH:(col_idx+1)*IMG_WIDTH] = img_scaled
        

    cv2.imwrite(filename, tiled_image)



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Directory to where pairwise matrices were saved (default directory will be called xyz_summary, where xyz refers to name of original mrc file)", required=True)
    parser.add_argument("--num_clusters", type=int, help="If you want to specify the number of clusters to group the class averages in, you can specify that here. Otherwise, the program will decide on the optimal number of clusters according to the silhouette distance") 
    args = parser.parse_args()

    input_dir = args.input_dir
    input_dir = input_dir[:-1] if input_dir.endswith('/') else input_dir

    if args.num_clusters is not None:
        if args.num_clusters <= 0:
            sys.exit('invalid input - num clusters must be positive')

    filepath_txt_file = '%s/filepath.txt' % input_dir 
    if Path(filepath_txt_file).exists() == False:
        sys.exit('invalid input - input_dir must be the directory where pairwise matrices were saved')

    image_2d_matrix = get_image_2d_matrix(input_dir)   
    particle_count_dict = get_particle_count(input_dir)

    print('calculating cluster labels')
    calc_spectral_cluster_labels(input_dir, 'corr', args.num_clusters)
    print('plotting distance matrices')
    plot_dist_matrix(input_dir, 'corr', 'spectral_clustering')
    plot_hierarchical_cluster(input_dir, 'edge', 'hierarchical_clustering')
    print('generating sorted mrc panel per cluster')
    gen_image_panel(input_dir, image_2d_matrix, particle_count_dict)

    print('generating histograms')
    hist_wrapper(input_dir)
    


