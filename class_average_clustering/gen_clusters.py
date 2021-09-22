import numpy as np
import matplotlib.pyplot as plt
import mrcfile
from sklearn import cluster
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from pathlib import Path
import sys
import collections 


def calc_spectral_cluster_labels(dist_matrix_path, dist_type, num_clusters=None):

    dist_matrix = np.genfromtxt('%s/%s_dist_matrix.csv' % (dist_matrix_path, dist_type), delimiter=',') 
    
    print(dist_matrix)

    if dist_type == 'corr':
        spectral_dist_matrix = 1-dist_matrix
    else:
        spectral_dist_matrix = dist_matrix
        spectral_dist_matrix = np.exp(-spectral_dist_matrix ** 2 / (2. * delta ** 2))

    print(spectral_dist_matrix)

    #cluster_labels_df = np.zeros((len(eps_array), dist_matrix.shape[0]+1))

    spectral_model_dict = {}
    spectral_model_sil_score = {}    
    
    optimal_cluster = ''
    max_sil_score = -1
    eps = .06

    if num_clusters is not None:

        clustering = cluster.SpectralClustering(n_clusters=num_clusters, n_init = 100, affinity='precomputed', assign_labels='kmeans').fit(spectral_dist_matrix)
        sil_score = metrics.silhouette_score(dist_matrix, clustering.labels_, metric='precomputed')
        spectral_model_sil_score[num_clusters] = sil_score
        optimal_cluster = clustering

      
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

        print(sil_score)
        print(clustering.labels_)

    cluster_output_folder = '%s/spectral' % dist_matrix_path
    Path(cluster_output_folder).mkdir(parents=True, exist_ok=True)
    np.savetxt('%s/%s_cluster_labels.csv' % (cluster_output_folder, dist_type), optimal_cluster.labels_) 
    np.savetxt('%s/%s_silhouette_score.csv' % (cluster_output_folder, dist_type), np.array([max_sil_score]))

    

     
for filename in ['KaiC_class_averages.mrc', 'cryosparc_P50_J436_020_class_averages.mrc', 'hLonOpenClosed_class_averages.mrc', 'cryosparc_P127_J4_020_class_averages.mrc']:

    print(filename)

    filename_wo_type = filename.split('.')[0]

    #dist_matrix_path = './output/%s/blur' % filename_wo_type
    #calc_spectral_cluster_labels(dist_matrix_path, 'corr')

    dist_matrix_path = './output/%s/clean' % filename_wo_type
    calc_spectral_cluster_labels(dist_matrix_path, 'corr')



