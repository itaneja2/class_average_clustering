class\_average\_clustering
==============================

Generates clusters, sorts class averages, and generates relevant input for histogram visualization. 


### Installation


1. Clone the git repository to your local machine and move into that directory

		git clone https://github.com/itaneja2/class_average_clustering.git
		cd class_average_clustering

3. Install class\_average\_clustering using `pip`

		pip install .

4. Then, whenever you want to check for updates simply run

		git pull
		pip install .
		
	The first command (`git pull`) checks the [git repo](https://github.com/itaneja2/class_average_clustering) for updates and then the second command installs the updated version.

### Usage

1.  Generate the input pairwise distance matrices:

		python gen_dist_matrix.py --mrc_file /path/to/mrc_file.mrc --star_file /path/to/star_file.star
	
	You can pass in additional arguments as follows: 

		python gen_dist_matrix.py --mrc_file /path/to/mrc_file.mrc --star_file /path/to/star_file.star --mirror 0 --scale .5 --ouptut_dir /path/to/folder/storing/output
		
	* By default, each image and its mirror image is used in the calculations of the distance matrices. If your class averages are generally symmetric (symmetric in the sense that you can flip the image, rotate it, and you will arrive at the original image), you can turn off the mirror calculation by setting `--mirror 0`. This will speed up the calculations by a factor 2. 
	* You have the option to downsample your image by a factor between 0 and 1 using the `--scale` option. This leads to a speedup in the calculations. Empirically, it appears individual class averages should have dimensions of at least 80 by 80. By default, images are not downsampled. 
	* You have the option to specify the directory where output will be saved. By default, this directory is set to the directory where the xyz.mrc file is stored. Specifically, folder is called xyz\_summary\_mirror=X\_scale=Y. 

2.  Generate the input for the web application:

		python plot_clusters.py --input_dir /path/to/folder/generated/by/gen_dist_matrix

	You can pass in additional arguments as follows: 

		python plot_clusters.py --input_dir /path/to/folder/generated/by/gen_dist_matrix --num_clusters 1
		
	* You have the option to specify the number of clusters generated. By default, the program separates the class averages out into the 'optimal' number of clusters based on similarities of their normalized cross-correlation. This appears to be useful step to separate junk from non-junk. However, this step may generate clusters with class averages from different views that visually appear very distinct. If you don't want to separate your class averages out into different clusters, you can specify this with the `--num_clusters 1` option. More generally, if you want to separate your class averages out into N clusters, you can specify this with the `--num_clusters N` option.
		
Once this is done running, you should see a file structure looking as follows:

###### File Structure
    50S_ribosome_summary_mirror=1_scale=1
    |
    +----> class_average_panel_plots
           |
           +----> cluster_0.mrc
           +----> cluster_1.mrc
    |
    +----> filepath.txt
    |
    +----> hierarchical_clustering
           |
           +----> 
    +----> histogram_plots
           |
           +----> 
    +----> input
           |
           +---->        
    +----> pairwise_matrix
           |
           +---->   
    +----> spectral_clustering
           |
           +---->  

`cluster_0.mrc` and `cluster_1.mrc` refer to the original class average file split into two subsets. Each cluster is also sorted, so similar images appear near each other in the output. 
