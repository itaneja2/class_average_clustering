import sys
import os 
import mrcfile 
import glob
import pickle5 as pickle
from collections import OrderedDict
from pathlib import Path

def remove_files_in_folder(folder):

    files = glob.glob('%s/*.*' % folder)
    for f in files:
        os.remove(f)

def save_obj(obj, name):

    if '.pkl' not in name:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):

    if '.pkl' not in name:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        with open(name, 'rb') as f:
            return pickle.load(f)

        
def sort_dict(dct):

    dct_copy = {}
    for key in sorted(dct.keys()):
        dct_copy[key] = dct[key]
    return(dct_copy)


def get_image_2d_matrix(input_dir):

    mrc_path = '%s/input/*.mrc' % input_dir
    mrc_file = glob.glob(mrc_path)
    
    if len(mrc_file) == 0:
        sys.exit('mrc file does not exist in %s')
    elif len(mrc_file) > 1:
        sys.exit('multiple mrc files exist in %s')

    mrc = mrcfile.open(mrc_file[0], mode='r')
    return(mrc.data)


def get_particle_count(input_dir):

    particle_count_path = '%s/input/*particle_counts.pkl' % input_dir
    particle_count_file = glob.glob(particle_count_path)
    
    if len(particle_count_file) == 0:
        sys.exit('particle count file does not exist in %s')
    elif len(particle_count_file) > 1:
        sys.exit('multiple particle count files exist in %s')

    particle_count_dict = load_obj(particle_count_file[0])
    return(particle_count_dict)


def get_particle_count_dict_cluster(particle_count_dict, image_list):

    particle_count_dict_cluster_c = {}
    for idx,i in enumerate(image_list):
        particle_count_dict_cluster_c[idx] = particle_count_dict[i]
    return(particle_count_dict_cluster_c)



