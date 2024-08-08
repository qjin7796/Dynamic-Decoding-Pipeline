import numpy as np, os, pickle, json
from nilearn import image
from datetime import datetime
from fastdist import fastdist
from scipy.stats import pearsonr


def read_data_file(filename):
    """ Read .npz, .pickle or .json file. 
        .npz for small nested dictionary, 
            always with allow_pickle=True, field name 'Results', 
            and .item() to unravel the dictionary.
        .json for small dictionary/list/np.ndarray.
        .pickle for large dictionary/list/np.ndarray.
    """
    if '.npz' in filename:
        try:
            data = np.load(filename, allow_pickle=True)['Results'].item()
        except:
            data = np.load(filename, allow_pickle=True)['Results']
    elif '.pickle' in filename:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    elif '.json' in filename:
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f'{filename} must be .npz, .pickle or .json file!')
    return data


def compute_searchlight_rdms(data_folder, subject_list, output_folder, metric, 
                             radius, searchlight_dict_file, mask_file, voxel_dict_file, 
                             space_str, smooth_str, confounds_str, drift_str, hrf_str, image_str, output_name_str):
    """ Compute subject / group mean searchlight rdms for every voxel in voxel_dict
        voxel_dict: {voxel_index: {'xyz': [x, y, z]}}
        mask: brain mask image file
        metric: distance metric, 'correlation', 'euclidean', 'cosine'
        radius: searchlight radius in mm, note that searchlight_dict is defined for human brain, 
            5mm equal to 3mm for monkey brain, 3mm equal to 2mm.
        output_name_str: monkey-3-subjects, human-27-subjects
        space_str: 'MNI152', 'Tom', 'D99'
        image_str: 'concat-zmaps'
        smooth_str: 'smooth-3mm', 'raw'
    """
    n_conditions = 144
    dist_vector_length = int(n_conditions * (n_conditions - 1) / 2)  # 10296
    # Read brain mask
    mask_data = image.get_data(mask_file)
    # Read voxel dict
    voxel_dict = read_data_file(voxel_dict_file)
    n_voxels = len(voxel_dict)
    print(f'{datetime.now()} {n_voxels} voxels in the mask.')
    # Read input images (lss-glm output)
    all_subject_input_imgs = {}
    all_voxel_coords_indices = {}
    for subject in subject_list:
        input_img_path = os.path.join(
            data_folder, subject, 
            f'LSS-GLM_{subject}_{smooth_str}_{confounds_str}_drift-{drift_str}_hrf-{hrf_str}_hpf-128_{image_str}.nii.gz'
        )
        input_img_data = image.get_data(input_img_path)
        all_subject_input_imgs[subject] = np.zeros((len(voxel_dict), n_conditions), dtype=np.float32)
        for voxel_i, voxel_info in voxel_dict.items():
            all_subject_input_imgs[subject][voxel_i, :] = input_img_data[voxel_info['xyz'][0], voxel_info['xyz'][1], voxel_info['xyz'][2], :]
            all_voxel_coords_indices[(voxel_info['xyz'][0], voxel_info['xyz'][1], voxel_info['xyz'][2])] = voxel_i
    # Read searchlight dict
    searchlight_dict = read_data_file(searchlight_dict_file)
    searchlight_array = searchlight_dict[radius]
    # Initialize output
    if radius == 5:
        radius_str = '19voxels'
    else:
        raise ValueError(f'radius {radius} not supported!')
    analysis_name = f'PAO_{output_name_str}_{space_str}_{smooth_str}_{confounds_str}_drift-{drift_str}_hrf-{hrf_str}_searchlight-{radius_str}_reuclidms'
    output_subfolder = os.path.join(output_folder, analysis_name)
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)
    log_file = os.path.join(output_subfolder, 'log.txt')
    with open(log_file, 'w') as f:
        ## Append datetime.now() followed by the analysis_params dict
        f.write(f'{datetime.now()}Starting {analysis_name}...\n')
    
    group_searchlight_rdms = np.zeros((n_voxels, dist_vector_length), dtype=np.float32)  # (n_voxels, 10296) np.float32
    for subject, subject_map in all_subject_input_imgs.items():
        subject_map = subject_map.astype(np.float32)  # (n_voxels, 144)
        all_voxel_searchlight_indices = {}
        for voxel_i, voxel_info in voxel_dict.items():
            voxel_x, voxel_y, voxel_z = voxel_info['xyz']
            ## Check if voxels in searchlight are valid
            voxel_searchlight_indices = []
            for searchlight_xyz in searchlight_array:
                searchlight_coords = np.array([voxel_x, voxel_y, voxel_z]) + searchlight_xyz
                try:
                    res = mask_data[tuple(searchlight_coords)]
                except:
                    res = 0
                if res > 0:
                    ### Further check if voxel has valid number in all conditions in subject_map
                    searchlight_voxel_i = all_voxel_coords_indices[tuple(searchlight_coords)]
                    if not np.all(subject_map[searchlight_voxel_i, :] == 0):
                        voxel_searchlight_indices.append(searchlight_voxel_i)
                ## Check if there are more than 3 voxels in the searchlight
            if len(voxel_searchlight_indices) > 2:
                all_voxel_searchlight_indices[voxel_i] = tuple(voxel_searchlight_indices)
        subject_rdms = np.zeros((len(voxel_dict), dist_vector_length), dtype=np.float32)  # (n_voxels, 10296)
        for voxel_i, voxel_searchlight_indices in all_voxel_searchlight_indices.items():
            voxel_searchlight_map = subject_map[voxel_searchlight_indices, :].transpose()  # (144, n_neighbor_voxels) np.float32
            search_size = voxel_searchlight_map.shape[1]
            ## Compute pairwise distance between conditions
            try:
                if metric == 'euclidean':
                    search_pdist_vector = fastdist.matrix_pairwise_distance(voxel_searchlight_map, fastdist.euclidean, metric, return_matrix=False) / np.sqrt(search_size)  # (10296,) np.float32
                else:
                    raise ValueError(f'{metric} not supported!')
            except:
                print(f'{datetime.now()} fastdist failed for {subject} {voxel_i} {voxel_searchlight_indices}')
                with open(log_file, 'a') as f:
                    f.write(f'{datetime.now()} fastdist failed for {subject} {voxel_i} {voxel_searchlight_indices}\n')
            if np.isnan(search_pdist_vector).any():
                print(f'{datetime.now()} fastdist returned nan for {subject} {voxel_i} {voxel_searchlight_indices}')
                with open(log_file, 'a') as f:
                    f.write(f'{datetime.now()} fastdist returned nan for {subject} {voxel_i} {voxel_searchlight_indices}\n')
            subject_rdms[voxel_i, :] = search_pdist_vector.squeeze()  # (10296,) np.float32
            if voxel_i % 30000 == 0:
                print(f'{datetime.now()} Searchlight RDM computed for {subject} voxel {voxel_i}...')
                print(f'-- mean {np.mean(subject_rdms[voxel_i, :])}, std {np.std(subject_rdms[voxel_i, :])}, max {np.max(subject_rdms[voxel_i, :])}, min {np.min(subject_rdms[voxel_i, :])}')
                with open(log_file, 'a') as f:
                    f.write(f'{datetime.now()} Searchlight RDM computed for {subject} voxel {voxel_i}...\n')
                    f.write(f'-- mean {np.mean(subject_rdms[voxel_i, :])}, std {np.std(subject_rdms[voxel_i, :])}, max {np.max(subject_rdms[voxel_i, :])}, min {np.min(subject_rdms[voxel_i, :])}\n')
        ## Compute mean searchlight rdm for each voxel
        if len(subject_list) > 1:
            group_searchlight_rdms += subject_rdms / len(subject_list)
            ## Save subject_rdm to sub output folder
            subject_rdm_output_file = os.path.join(output_subfolder, f"{analysis_name.replace(output_name_str, subject)}.pickle")
            with open(subject_rdm_output_file, 'wb') as f:
                pickle.dump(subject_rdms, f)
            with open(log_file, 'a') as f:
                f.write(f'{datetime.now()} Searchlight RDM computed for {subject} saved to {subject_rdm_output_file}\n')
        else:
            ## Save subject_rdm to main output folder
            subject_rdm_output_file = os.path.join(output_folder, f"{analysis_name.replace(output_name_str, subject)}.pickle")
            with open(subject_rdm_output_file, 'wb') as f:
                pickle.dump(subject_rdms, f)
            with open(log_file, 'a') as f:
                f.write(f'{datetime.now()} Searchlight RDM computed for {subject} saved to {subject_rdm_output_file}\n')
    ## Save group_searchlight_rdms to main output folder
    if len(subject_list) > 1:
        group_searchlight_rdms_output_file = os.path.join(output_folder, f"{analysis_name}.pickle")
        with open(group_searchlight_rdms_output_file, 'wb') as f:
            pickle.dump(group_searchlight_rdms, f)
        with open(log_file, 'a') as f:
            f.write(f'{datetime.now()} Group searchlight RDM saved to {group_searchlight_rdms_output_file}\n')
        return group_searchlight_rdms
    else:
        with open(log_file, 'a') as f:
            f.write(f'{datetime.now()} Only one subject, group searchlight RDM not saved.\n')
        return subject_rdms


my_data_folder = '/scratch/leuven/347/vsc34741/code/rsa_preprocessing/lss-glm_D99-raw-spm1-mion-hmc-cosine'
my_subject_list = ['sub-Radja', 'sub-Dobby', 'sub-Elmo']
my_output_name_str = 'monkey-3-subjects'
my_output_folder = '/scratch/leuven/347/vsc34741/code/rsa_preprocessing/rdms'
os.makedirs(my_output_folder, exist_ok=True)
my_hrf_str = 'mion'
my_space_str = 'D99'
my_mask_file = '/data/leuven/347/vsc34741/code/rsa_preprocessing/D99_template_1mm_mask_lr.nii'
my_voxel_dict_file = '/data/leuven/347/vsc34741/code/rsa_preprocessing/D99_voxel_indexing_dict.npz'

my_metric = 'euclidean'
my_radius = 5
my_searchlight_dict_file = '/data/leuven/347/vsc34741/code/rsa_preprocessing/PAO_human_searchlight_dict.npz'
my_smooth_str = 'raw'
my_confounds_str = '6-motion-confounds'
my_drift_str = 'cosine'
my_image_str = 'concat-zmaps'

_ = compute_searchlight_rdms(my_data_folder, my_subject_list, my_output_folder, my_metric,
                             my_radius, my_searchlight_dict_file, my_mask_file, my_voxel_dict_file,
                             my_space_str, my_smooth_str, my_confounds_str, my_drift_str, my_hrf_str, my_image_str, my_output_name_str)
