import os, pickle
import numpy as np
from nilearn import image
from statsmodels.stats.multitest import fdrcorrection
from scipy.ndimage import generate_binary_structure, label


voxel_indexing_file = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/rsa_082023/data/D99_voxel-dict.npz'
# voxel_indexing_file = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/rsa_082023/data/PAO_human-27-subjects_refined-voxel-dict.npz'
voxel_indexing = np.load(voxel_indexing_file, allow_pickle=True)['Results'].item()
output_folder = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/rsa_082023/results/output_D99_refined_batch-3_monkey-3-r-3-ed_RSA-regout-v2-3D'
# output_folder = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/rsa_082023/results/output_refined_batch-3_monkey-3-r-3-ed_RSA-1/'
# output_folder = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/rsa_082023/results/output_PAO_monkey-3-r-3-ed_AT-OHSD-global3D_ridge-regression/'
# output_folder = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/rsa_082023/results/output_refined_batch-1_human-27-3-5-ed_RSA-1/'
# output_folder = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/rsa_082023/results/output_PAO_human-27-r-5-ed_AT-OHSD-global3D_ridge-regression/'
# output_folder = '/data/local/u0151613/Qiuhan_Projects/Encoding_Qiuhan/rsa_082023/results/output_refined_batch-1_human-27-r-5-ed_RSA-regout-v2/'
output_prefix = 'PAO_monkey-3-r-3-ed_'
# output_prefix = 'PAO_human-27-r-5-ed_'
# output_prefix = ''

# n_permutations = 10000
# fdr_alpha = 10 / n_permutations
# q_thr = 100 / n_permutations
z_thr_list = [1.96]  # , 2.3, 2.58, 3.0
z_thr_str_list = ['_95ci']  # , '_23ci', '_99ci', '_999ci']

# Cluster extent thresholding
cluster_correction = True
cluster_forming_struct = generate_binary_structure(3, 1)  # 6-connectivity
cluster_extent_type = 'size'  # 'mass' = sum of statistic or 'size' = num of voxels
cluster_mass_threshold_list = [95]  # n = n% confidence interval 
cluster_size_threshold_list = [20]  # n = n voxels


for file in os.listdir(output_folder):
    if file.endswith('.nii.gz'):
        # if 'standard-rsa' in file or 'ggrad' in file:
        # if 'action-type_regout-gpitmv-gmetmv-ohsd' in file:
        print(file)
        # Load zmap
        input_zmap_image = image.load_img(os.path.join(output_folder, file))
        input_zmap_data = input_zmap_image.get_fdata()
        # Get vector
        input_zvector = np.zeros(len(voxel_indexing))
        for voxel_i, voxel_info in voxel_indexing.items():
            x, y, z = voxel_info['xyz']
            input_zvector[voxel_i] = input_zmap_data[x, y, z]
        # Normalize
        print('Input z vector stats:', np.nanmean(input_zvector), np.nanstd(input_zvector))
        output_zvector = (input_zvector - np.nanmean(input_zvector)) / np.nanstd(input_zvector)
        print('Output z vector stats:', np.nanmean(output_zvector), np.nanstd(output_zvector))
        # Back to zmap
        zmap_data = np.zeros(input_zmap_data.shape)
        for voxel_i, voxel_info in voxel_indexing.items():
            x, y, z = voxel_info['xyz']
            zmap_data[x, y, z] = output_zvector[voxel_i]
        zmap_image = image.new_img_like(input_zmap_image, zmap_data)
        if not os.path.exists(os.path.join(output_folder, 'corrected')):
            os.mkdir(os.path.join(output_folder, 'corrected'))
        zmap_image.to_filename(os.path.join(output_folder, 'corrected', output_prefix + file[:-7] + '_uncorrected.nii.gz'))
        print('Unthresholded zmap saved to', os.path.join(output_folder, 'corrected', output_prefix + file[:-7] + '_uncorrected.nii.gz'))
        # Threshold zmap
        for z_i, z_thr in enumerate(z_thr_list):
            z_thr_str = z_thr_str_list[z_i]
            z_thr_map_data = zmap_data.copy()
            z_thr_map_data[np.abs(z_thr_map_data) < z_thr] = 0
            z_thr_map_image = image.new_img_like(input_zmap_image, z_thr_map_data)
            z_thr_map_image.to_filename(os.path.join(output_folder, 'corrected', output_prefix + file[:-7] + z_thr_str + '.nii.gz'))
            print('Thresholded zmap saved to', os.path.join(output_folder, 'corrected', output_prefix + file[:-7] + z_thr_str + '.nii.gz'))
            ## Cluster extent thresholding
            if cluster_correction:
                print('Cluster extent thresholding by cluster', cluster_extent_type)
                ### Form clusters
                all_clusters, n_clusters = label(z_thr_map_data, cluster_forming_struct)
                ### Count 0 voxels
                n_zeros = np.count_nonzero(all_clusters == 0)
                ### Compute cluster mass or size
                if cluster_extent_type == 'mass':
                    cluster_masses = np.zeros(n_clusters, dtype=float)
                    for i in range(n_clusters):
                        cluster_masses[i] = np.sum(z_thr_map_data[all_clusters == i+1])
                    ### Sort for percentile thresholding
                    sorted_cluster_masses = np.sort(cluster_masses)
                    for cluster_mass_threshold in cluster_mass_threshold_list:
                        try:
                            cluster_select_thr = sorted_cluster_masses[int(np.round(cluster_mass_threshold / 100 * n_clusters))]
                        except:
                            cluster_select_thr = sorted_cluster_masses[int(np.round(cluster_mass_threshold / 100 * n_clusters)) - 1]
                        ### Mass thresholding
                        cluster_thresholded_zmap_data = np.copy(z_thr_map_data)
                        for i in range(n_clusters):
                            if cluster_masses[i] < cluster_select_thr:
                                cluster_thresholded_zmap_data[all_clusters == i+1] = 0
                        cluster_thresholded_zmap_image = image.new_img_like(input_zmap_image, cluster_thresholded_zmap_data)
                        cluster_thresholded_zmap_image.to_filename(os.path.join(output_folder, 'corrected', output_prefix + file[:-7] + z_thr_str + f'_cluster-{cluster_extent_type}-' + str(cluster_mass_threshold) + '.nii.gz'))
                elif cluster_extent_type == 'size':
                    cluster_sizes = np.zeros(n_clusters, dtype=int)
                    for i in range(n_clusters):
                        cluster_sizes[i] = np.count_nonzero(all_clusters == i+1)
                    ### Size thresholding
                    for cluster_size_threshold in cluster_size_threshold_list:
                        cluster_thresholded_zmap_data = np.copy(z_thr_map_data)
                        num_left_clusters = 0
                        for i in range(n_clusters):
                            if cluster_sizes[i] < cluster_size_threshold:
                                cluster_thresholded_zmap_data[all_clusters == i+1] = 0
                        cluster_thresholded_zmap_image = image.new_img_like(input_zmap_image, cluster_thresholded_zmap_data)
                        cluster_thresholded_zmap_image.to_filename(os.path.join(output_folder, 'corrected', output_prefix + file[:-7] + z_thr_str + f'_cluster-{cluster_extent_type}-' + str(cluster_size_threshold) + '.nii.gz'))
                        ### Check and save cluster mask
                        num_left_clusters = 0
                        all_clusters_1, n_clusters_1 = label(cluster_thresholded_zmap_data, cluster_forming_struct)
                        for i in range(n_clusters_1):
                            if np.sum(cluster_thresholded_zmap_data[all_clusters_1 == i+1]) > 0:
                                num_left_clusters += 1
                                cluster_mask_data = np.zeros(cluster_thresholded_zmap_data.shape)
                                cluster_mask_data[all_clusters_1 == i+1] = 1
                                # cluster_mask_image = image.new_img_like(input_zmap_image, cluster_mask_data)
                                # cluster_mask_image.to_filename(os.path.join(output_folder, 'corrected', output_prefix + file[:-7] + z_thr_str + f'_cluster-{cluster_extent_type}-' + str(cluster_size_threshold) + f'_cluster-{i+1}.nii.gz'))
                        print('Cluster size threshold:', cluster_size_threshold, 'Num clusters:', num_left_clusters)
                        print('Cluster size thresholded zmap saved to', os.path.join(output_folder, 'corrected', output_prefix + file[:-7] + z_thr_str + f'_cluster-{cluster_extent_type}-' + str(cluster_size_threshold) + '.nii.gz'))
                else:
                    raise ValueError('cluster_extent_type must be either "mass" or "size"')



# for analysis_type, analysis_result in result.items():
#     for model_name, model_result in analysis_result.items():
#         print(analysis_type, model_name)
#         # Type 1
#         model_uncorrected_zmap = model_result['uncorrected_zmap'].get_fdata()
#         model_uncorrected_rmap = model_result['uncorrected_rmap'].get_fdata()
#         # Get vector
#         uz_vector = np.zeros(len(voxel_indexing))
#         # ur_vector = np.zeros(len(voxel_indexing))
#         voxel_std_inv = np.sqrt(len(voxel_indexing)-3)
#         for voxel_i, voxel_info in voxel_indexing.items():
#             x, y, z = voxel_info['xyz']
#             uz_vector[voxel_i] = model_uncorrected_zmap[x, y, z]
#             # ur_vector[voxel_i] = np.arctanh(model_uncorrected_rmap[x, y, z])
#         # Rescale
#         # ur_vector_std_inv = 1 / np.nanstd(ur_vector)
#         # uz_vector = uz_vector * voxel_std_inv / ur_vector_std_inv
#         # Normalize
#         print(np.nanmean(uz_vector), np.nanstd(uz_vector))
#         uz_vector = (uz_vector - np.nanmean(uz_vector)) / np.nanstd(uz_vector)
#         print(np.nanmean(uz_vector), np.nanstd(uz_vector))
#         # Back to zmap
#         zmap_data = np.zeros(model_uncorrected_zmap.shape)
#         for voxel_i, voxel_info in voxel_indexing.items():
#             x, y, z = voxel_info['xyz']
#             zmap_data[x, y, z] = uz_vector[voxel_i]
#         # Save brain map z thresholded at 1.96, 2.3, 2.58
#         # zmap_image = image.new_img_like(model_result['uncorrected_zmap'], zmap_data)
#         # zmap_image.to_filename(os.path.join(output_folder, model_name + '.nii.gz'))
#         zmap_data_95 = zmap_data.copy()
#         zmap_data_95[np.abs(zmap_data_95) < 1.96] = 0
#         zmap_data_mid = zmap_data.copy()
#         zmap_data_mid[np.abs(zmap_data_mid) < 2.3] = 0
#         zmap_data_99 = zmap_data.copy()
#         zmap_data_99[np.abs(zmap_data_99) < 2.58] = 0
#         zmap_image_95 = image.new_img_like(model_result['uncorrected_zmap'], zmap_data_95)
#         zmap_image_mid = image.new_img_like(model_result['uncorrected_zmap'], zmap_data_mid)
#         zmap_image_99 = image.new_img_like(model_result['uncorrected_zmap'], zmap_data_99)
#         zmap_image_95.to_filename(os.path.join(output_folder, model_name + '_95ci.nii.gz'))
#         zmap_image_mid.to_filename(os.path.join(output_folder, model_name + '_midci.nii.gz'))
#         zmap_image_99.to_filename(os.path.join(output_folder, model_name + '_99ci.nii.gz'))


#         # # Type 2
#         # model_uncorrected_zmap = model_result['uncorrected_zmap'].get_fdata()
#         # model_uncorrected_pmap = model_result['uncorrected_pmap'].get_fdata()
#         # uz_vector = np.zeros(len(voxel_indexing))
#         # up_vector = np.zeros(len(voxel_indexing))
#         # for voxel_i, voxel_info in voxel_indexing.items():
#         #     x, y, z = voxel_info['xyz']
#         #     uz_vector[voxel_i] = model_uncorrected_zmap[x, y, z]
#         #     up_vector[voxel_i] = model_uncorrected_pmap[x, y, z]
#         # # FDR control
#         # _, q_vector = fdrcorrection(up_vector, alpha=fdr_alpha)
#         # # Threshold zmap
#         # z_vector = uz_vector.copy()
#         # uz_vector[q_vector > q_thr] = 0
#         # # Normalize z_vector to estimate confidence interval
#         # z_mean = np.nanmean(uz_vector)
#         # z_std = np.nanstd(uz_vector)
#         # # Transform z_vector to normal distribution
#         # uz_vector = (uz_vector - z_mean) / z_std
#         # # Threshold 95% confidence interval
#         # # z_vector[np.abs(uz_vector) < 1.96] = 0
#         # # Threshold 99% confidence interval
#         # z_vector[np.abs(uz_vector) < 3] = 0
#         # # Transform back to brain map
#         # zmap_data = np.zeros(model_uncorrected_zmap.shape)
#         # for voxel_i, voxel_info in voxel_indexing.items():
#         #     x, y, z = voxel_info['xyz']
#         #     zmap_data[x, y, z] = z_vector[voxel_i]
#         # # Save brain map
#         # zmap_image = image.new_img_like(model_result['uncorrected_zmap'], zmap_data)
#         # zmap_image.to_filename(os.path.join(output_folder, model_name + '_3.nii.gz'))
