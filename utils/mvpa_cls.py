# NB: Run on single GPU only.


from utils_rsa import *
import argparse


# Argument: analysis_params, a dictionary containing the following fields:
# (1) 'data_folder': str,
#     data_folder contains all files needed for running the analysis are stored;
#     these data are shared by jobs.
# (2) 'analysis_name': str,
#     name for output folder and log file.
# (3) 'rsa_model_sets': dict,
#     a dictionary indicating RSA analysis type and model list to use.
#     {'standard': [model_name_1, model_name_2, ...]}
#     model_name can only be one of ['action-type', 'viewpoint', 'object-type', 'handedness', 'experimenter']
#     {'partial': {model_set_name_1: [model_name_1, model_name_2, ...]}, ...}
#     model_set_name can be any str, model_name = seg_name + '-' + feature_name.
#     seg_name_list = ['global', 'arm', 'hand', 'object-hand', 'object-arm', 'object', 'face']
# (4) 'permutation_indices_filename': str,
#     (10000, 144) np.int8 array
# (5) 'brain_mask_filename': str,
#     (x, y, z) int8 NIfTI image.
# (6) 'voxel_indexing_dict_filename': str,
#     {0: {'xyz': (x, y, z), 'index': k}, 1, 2, ...}
# (7) 'group_searchlight_rdms_filename': str,
#     filename for the group searchlight RDMs (n_voxels, 10296) np.float16 array.
# (8) n_permutations: int,
#     number of permutations for permutation test, not exceeding 10000.
#      The analysis speed depends on n_voxels and n_permutations.
#          10000 permutations on 78000 voxels takes about 8~13 mins on a wice GPU (1 batch),
#          and 8~13 * n_batches mins on a genius GPU (typically 3 batches).
# (9) n_batches: int,
#     number of batches for reducing memory requirement of the brain rdms.
#     For genius GPU, 3 batches for human data and 4 batches for monkey data.
#     For wice GPU, 1 batch is fine.
# (10) fdr_alpha_list: a list of float,
#      alpha for FDR correction.
# (11) q_threshold_list: a list of float,
#      threshold for corrected permutation test significance.
# (12) smooth_fwhm_list: a list of float,
#      FWHM for smoothing permutation test significant zmaps.


def read_data_file(folder, filename):
    """ Read .npz, .pickle or .json file. 
        .npz for small nested dictionary, 
            always with allow_pickle=True, field name 'Results', 
            and .item() to unravel the dictionary.
        .json for small dictionary/list/np.ndarray.
        .pickle for large dictionary/list/np.ndarray.
    """
    if '.npz' in filename:
        try:
            data = np.load(os.path.join(folder, filename), allow_pickle=True)['Results'].item()
        except:
            data = np.load(os.path.join(folder, filename), allow_pickle=True)['Results']
    elif '.pickle' in filename:
        with open(os.path.join(folder, filename), 'rb') as f:
            data = pickle.load(f)
    elif '.json' in filename:
        with open(os.path.join(folder, filename), 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f'{filename} must be .npz, .pickle or .json file!')
    return data


# (1) Get arguments (all arguments wrapped in analysis_params dict)
parser = argparse.ArgumentParser()
parser.add_argument('--analysis_params', type=str)  # a dict of parameters needed for analysis
args = parser.parse_args()
analysis_params_path = args.analysis_params
with open(analysis_params_path, 'r') as f:
    analysis_params = json.load(f)
# Assert all arguments 
data_folder = analysis_params['data_folder']
assert os.path.exists(data_folder), 'data_folder does not exist!'
analysis_name = analysis_params['analysis_name']
analysis_prefix = analysis_name.split('_19voxels')[0]
## Check if 'glm' smoothing param and 'searchlight' radius param are specified in analysis_name
if 'glm' not in analysis_name:
    print('Warning: "glm" smoothing param not specified in analysis_name!')
if 'searchlight' not in analysis_name:
    print('Warning: "searchlight" radius param not specified in analysis_name!')
## Check if output folder for this analysis_name already exists
output_folder = os.path.join(data_folder, f'output_{analysis_name}')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    # Warning: this will overwrite existing files!
    print(f'Warning: {output_folder} already exists! Overwriting existing files!')
rsa_model_sets = analysis_params['rsa_model_sets']
# rsa_model_sets must and only contain either or both of 'standard' and 'partial' keys.
assert isinstance(rsa_model_sets, dict), 'rsa_model_sets must be a dict!'
for key in list(rsa_model_sets.keys()):
    assert key in ['standard', 'partial'], 'rsa_model_sets key must be either "standard" or "partial"!'
n_permutations = analysis_params['n_permutations']
assert isinstance(n_permutations, int), 'n_permutations must be an int!'
n_batches = analysis_params['n_batches']
assert isinstance(n_batches, int), 'n_batches must be an int!'
fdr_alpha_list = analysis_params['fdr_alpha_list']
assert isinstance(fdr_alpha_list, list), 'fdr_alpha_list must be a list!'
q_threshold_list = analysis_params['q_threshold_list']
assert isinstance(q_threshold_list, list), 'q_threshold_list must be a list of float!'
z_percentile_list = analysis_params['z_percentile_list']
assert isinstance(z_percentile_list, list), 'z_percentile_list must be a list of integer!'
cluster_size_threshold_list = analysis_params['cluster_size_threshold_list']
assert isinstance(cluster_size_threshold_list, list), 'cluster_size_threshold_list must be a list of integer!'
cluster_forming_struct = generate_binary_structure(3, 1)  # 6-connectivity

# Check if CUDA is available
if not torch.cuda.is_available():
    print('\n\nCUDA is not available!\n\n')
rsa_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# (2) Read data
categorical_model_rdms_filename = analysis_params['categorical_model_rdms_filename']
categorical_model_rdms = read_data_file(data_folder, categorical_model_rdms_filename)
assert isinstance(categorical_model_rdms, dict), 'categorical_model_rdms must be a dict!'
real_valued_model_rdms_filename = analysis_params['real_valued_model_rdms_filename']
real_valued_model_rdms = read_data_file(data_folder, real_valued_model_rdms_filename)
assert isinstance(real_valued_model_rdms, dict), 'real_valued_model_rdms must be a dict!'
permutation_indices_filename = analysis_params['permutation_indices_filename']
permutation_indices = read_data_file(data_folder, permutation_indices_filename)
assert isinstance(permutation_indices, np.ndarray), 'permutation_indices must be a np.ndarray!'
voxel_indexing_dict_filename = analysis_params['voxel_indexing_dict_filename']
voxel_indexing_dict = read_data_file(data_folder, voxel_indexing_dict_filename)
assert isinstance(voxel_indexing_dict, dict), 'voxel_indexing_dict must be a dict!'
n_voxels = len(voxel_indexing_dict)
group_searchlight_rdms_filename = analysis_params['group_searchlight_rdms_filename']
if 'vsc34741' in group_searchlight_rdms_filename:
    group_searchlight_rdms = read_data_file('', group_searchlight_rdms_filename)  # (n_voxels, 10296) np.float16 array
else:
    group_searchlight_rdms = read_data_file(data_folder, group_searchlight_rdms_filename)  # (n_voxels, 10296) np.float16 array
brain_mask_filename = analysis_params['brain_mask_filename']
brain_mask_file = os.path.join(data_folder, brain_mask_filename)
assert os.path.exists(brain_mask_file), 'brain_mask.nii does not exist!'
brain_mask_image = image.load_img(os.path.join(data_folder, brain_mask_filename))
brain_mask_array = image.get_data(os.path.join(data_folder, brain_mask_filename))
brain_shape = brain_mask_array.shape
# Compute triangle indices of a 144*144 matrix
triu_ind = np.triu_indices(144, k=1)

# (3) Initialize log
log_file = os.path.join(output_folder, f'log_{analysis_name}.txt')
with open(log_file, 'w') as f:
    ## Append datetime.now() followed by the analysis_params dict
    f.write(f'{datetime.now()}\n')
    f.write(f'Analysis parameters: {analysis_params}\n\n')

# (4) Read brain RDMs
group_searchlight_rdms = group_searchlight_rdms.astype(np.float32)  # (n_voxels, 10296) np.float32 array
# Split group searchlight RDMs into n_batches folds
batched_brain_rdms_dict = {}
n_voxels_per_batch = int(n_voxels / n_batches)
for b_i in range(n_batches):
    if b_i == n_batches - 1:
        batched_brain_rdms_dict[b_i] = group_searchlight_rdms[b_i*n_voxels_per_batch:, :]
    else:
        batched_brain_rdms_dict[b_i] = group_searchlight_rdms[b_i*n_voxels_per_batch:(b_i+1)*n_voxels_per_batch, :]
with open(log_file, 'a') as f:
    f.write(f'{datetime.now()} Split group searchlight brain RDMs into {n_batches} folds.\n')


# (5) Run standard RSA
if 'standard' in rsa_model_sets:
    with open(log_file, 'a') as f:
        f.write(f'\n{datetime.now()} Running standard RSA...\n')
    ## Loop through models
    rsa_model_list = rsa_model_sets['standard']
    for model_set_name in rsa_model_list:
        if model_set_name in ['action-type', 'viewpoint', 'handedness', 'object-type', 'experimenter', 'at', 'vp', 'hd', 'ot']:
            with open(log_file, 'a') as f:
                f.write(f'{datetime.now()} Comparing with categorical {model_set_name}...\n')
            model_rdm_vector = categorical_model_rdms[model_set_name].astype(np.float32)  # (10296,) np.float32 array
        else:
            with open(log_file, 'a') as f:
                f.write(f'{datetime.now()} Comparing with real-valued {model_set_name}...\n')
            model_rdm_vector = real_valued_model_rdms[model_set_name].astype(np.float32)  # (10296,) np.float32 array
        ### Convert to squareform tensor
        model_rdm_tensor_sqaureform = get_tensor_squareform(model_rdm_vector, 144).to(rsa_device)  # (144, 144) torch.float32
        ### Loop through batches of voxel RDMs
        baseline_corr_list = []
        count_pass_baseline_list = []
        percent_pass_baseline_list = []
        for b_i in range(n_batches):
            brain_rdms_array = batched_brain_rdms_dict[b_i]
            ### Conver to squareform tensor
            brain_rdms_tensor_squareform = get_tensor_squareform(brain_rdms_array, 144).to(rsa_device)  # (n_voxels_per_batch, 144, 144) torch.float32
            ### Compute correlation (uncorrected)
            baseline_corr_tensor = compute_r(brain_rdms_tensor_squareform, model_rdm_tensor_sqaureform, triu_ind).cpu()  # (n_voxels_per_batch,) torch.float32
            baseline_corr_list.append(baseline_corr_tensor.numpy())  # (n_voxels_per_batch) list of np.float32 arrays
            ### Permutation test
            count_pass_baseline_tensor = torch.zeros_like(baseline_corr_tensor, dtype=torch.int8)  # (n_voxels_per_batch,) torch.int8
            for p_i in range(n_permutations):
                perm_ind = permutation_indices[p_i, :]
                if p_i % int(n_permutations/100) == 0:
                    with open(log_file, 'a') as f:
                        f.write(f'{datetime.now()}, Permutation {p_i}\n')
                perm_corr_tensor = compute_r(brain_rdms_tensor_squareform, model_rdm_tensor_sqaureform, triu_ind, perm_ind).cpu()
                ### Compare with baseline (num_voxels,) torch.float32
                add_count = perm_corr_tensor > baseline_corr_tensor
                add_count = add_count.type(torch.int8)
                count_pass_baseline_tensor += add_count
            percent_pass_baseline_tensor = count_pass_baseline_tensor / n_permutations  # (n_voxels_per_batch,) torch.float32
            count_pass_baseline_list.append(count_pass_baseline_tensor.numpy())  # (n_voxels_per_batch) list of np.int8 arrays
            percent_pass_baseline_list.append(percent_pass_baseline_tensor.numpy())  # (n_voxels_per_batch) list of np.float32 arrays
            with open(log_file, 'a') as f:
                f.write(f'{datetime.now()}, Batch {b_i} {n_permutations} permutations done.\n')
        ### Concatenate batch results
        baseline_corr = np.concatenate(baseline_corr_list, axis=0)  # (n_voxels,) np.float32 array
        count_pass_baseline = np.concatenate(count_pass_baseline_list, axis=0)  # (n_voxels,) np.int8 array
        percent_pass_baseline = np.concatenate(percent_pass_baseline_list, axis=0)  # (n_voxels,) np.float32 array
        ### Fisher transform
        uncorr_z_vector = np.arctanh(baseline_corr)  # (n_voxels,) np.float32 array
        uncorr_z_vector = (uncorr_z_vector - np.nanmean(uncorr_z_vector)) / np.nanstd(uncorr_z_vector)
        ### Plot on brain
        uncorrected_rmap_array = np.zeros_like(brain_mask_array, dtype=np.float32)  # brain_shape np.float32 array
        uncorrected_zmap_array = np.zeros_like(brain_mask_array, dtype=np.float32)  # brain_shape np.float32 array
        uncorrected_pmap_array = np.zeros_like(brain_mask_array, dtype=np.float32)  # brain_shape np.float32 array
        for voxel_i, voxel_info in voxel_indexing_dict.items():
            voxel_x, voxel_y, voxel_z = voxel_info['xyz']
            uncorrected_rmap_array[voxel_x, voxel_y, voxel_z] = baseline_corr[voxel_i]
            uncorrected_zmap_array[voxel_x, voxel_y, voxel_z] = uncorr_z_vector[voxel_i]
            uncorrected_pmap_array[voxel_x, voxel_y, voxel_z] = percent_pass_baseline[voxel_i]
        uncorrected_rmap_img = image.new_img_like(brain_mask_image, uncorrected_rmap_array)  # brain_shape image
        uncorrected_zmap_img = image.new_img_like(brain_mask_image, uncorrected_zmap_array)  # brain_shape image
        uncorrected_pmap_img = image.new_img_like(brain_mask_image, uncorrected_pmap_array)  # brain_shape image
        uncorrected_rmap_img.to_filename(os.path.join(output_folder, f'{analysis_prefix}_standard-rsa_{model_set_name}_uncorrected-rmap.nii.gz'))
        uncorrected_zmap_img.to_filename(os.path.join(output_folder, f'{analysis_prefix}_standard-rsa_{model_set_name}_uncorrected-zmap.nii.gz'))
        uncorrected_pmap_img.to_filename(os.path.join(output_folder, f'{analysis_prefix}_standard-rsa_{model_set_name}_uncorrected-pmap.nii.gz'))
        ## FDR control
        for fdr_alpha in fdr_alpha_list:
            ### FDR controlled zmap
            rejected, q_vector = fdrcorrection(percent_pass_baseline, alpha=fdr_alpha, method='indep', is_sorted=False)  # (n_voxels,) float np.array
            for q_threshold in q_threshold_list:
                fdr_controlled_q_vector = q_vector <= q_threshold  # (n_voxels,) bool np.array
                fdr_controlled_z_vector = uncorr_z_vector * fdr_controlled_q_vector.astype(np.int8)  # (n_voxels,) np.float32 array
                ### Normalize
                # fdr_controlled_z_vector = (fdr_controlled_z_vector - np.nanmean(fdr_controlled_z_vector)) / np.nanstd(fdr_controlled_z_vector)
                ### Construct brain array
                fdr_controlled_zmap_array = np.zeros_like(brain_mask_array, dtype=np.float32)  # brain_shape np.float32 array
                for voxel_i, voxel_info in voxel_indexing_dict.items():
                    voxel_x, voxel_y, voxel_z = voxel_info['xyz']
                    fdr_controlled_zmap_array[voxel_x, voxel_y, voxel_z] = fdr_controlled_z_vector[voxel_i]
                fdr_controlled_zmap_img = image.new_img_like(brain_mask_image, fdr_controlled_zmap_array)  # brain_shape image
                fdr_controlled_zmap_img.to_filename(os.path.join(output_folder, f'{analysis_prefix}_partial-rsa_{model_set_name}_fdr-{fdr_alpha}_q-{q_threshold}.nii.gz'))
                ### Cluster correction: apply 95% threshold
                for z_percentile in z_percentile_list:
                    z_thr_array = fdr_controlled_zmap_array.copy()
                    if z_percentile == 95:
                        z_threshold = 1.96
                    elif z_percentile == 99:
                        z_threshold = 2.58
                    else:
                        two_sided_percentile = 50 + z_percentile/2
                        z_threshold = np.percentile(fdr_controlled_z_vector, two_sided_percentile)
                    z_thr_array[np.abs(z_thr_array) < z_threshold] = 0
                    #### Form clusters
                    all_clusters, n_clusters = label(z_thr_array, cluster_forming_struct)
                    n_zeros = np.count_nonzero(all_clusters == 0)
                    cluster_sizes = np.zeros(n_clusters, dtype=int)
                    for i in range(n_clusters):
                        cluster_sizes[i] = np.count_nonzero(all_clusters == i+1)
                    #### Cluster thresholding
                    for cluster_size_threshold in cluster_size_threshold_list:
                        cluster_corrected_zmap_array = z_thr_array.copy()
                        num_left_clusters = 0
                        for i in range(n_clusters):
                            if cluster_sizes[i] < cluster_size_threshold:
                                cluster_corrected_zmap_array[all_clusters == i+1] = 0
                        #### Save
                        cluster_corrected_zmap_img = image.new_img_like(brain_mask_image, cluster_corrected_zmap_array)  # brain_shape image
                        cluster_corrected_zmap_img.to_filename(os.path.join(output_folder, f'{analysis_prefix}_standard-rsa_{model_set_name}_fdr-{fdr_alpha}_q-{q_threshold}_z-{z_percentile}_cluster-size-{cluster_size_threshold}.nii.gz'))
    with open(log_file, 'a') as f:
        f.write(f'{datetime.now()} Done.\n\n')


# (6) Run partial RSA
if 'partial' in rsa_model_sets:
    with open(log_file, 'a') as f:
        f.write(f'\n{datetime.now()} Running partial RSA...\n')
    ## Loop through model sets
    for model_set_name, rsa_model_list in rsa_model_sets['partial'].items():
        with open(log_file, 'a') as f:
            f.write(f'{datetime.now()} Comparing with {model_set_name}...\n')
        ## First model is the target model, the rest are control models to be regressed out
        target_model_name = rsa_model_list[0]
        if target_model_name in ['action-type', 'viewpoint', 'handedness', 'object-type', 'experimenter', 'at', 'vp', 'hd', 'ot']:
            with open(log_file, 'a') as f:
                f.write(f'{datetime.now()} Target model is categorical {target_model_name}...\n')
            target_model_rdm_vector = categorical_model_rdms[target_model_name].astype(np.float32)  # (10296,) np.float32 array
        else:
            with open(log_file, 'a') as f:
                f.write(f'{datetime.now()} Target model is real-valued {target_model_name}...\n')
            target_model_rdm_vector = real_valued_model_rdms[target_model_name].astype(np.float32)  # (10296,) np.float32 array
        target_model_rdm_tensor = torch.from_numpy(target_model_rdm_vector)  # (10296,) torch.float32
        ## First model is the target model, the rest are control models to be regressed out
        control_model_rdms_vector = np.zeros((len(rsa_model_list)-1, 10296), dtype=np.float32)  # (n_controll_models, 10296) np.float32 array
        for model_i in range(1, len(rsa_model_list)):
            model_name = rsa_model_list[model_i]
            if model_name in ['action-type', 'viewpoint', 'handedness', 'object-type', 'experimenter', 'at', 'vp', 'hd', 'ot']:
                model_rdm_vector = categorical_model_rdms[model_name].astype(np.float32)  # (10296,) np.float32 array
            else:
                model_rdm_vector = real_valued_model_rdms[model_name].astype(np.float32)  # (10296,) np.float32 array
            control_model_rdms_vector[model_i-1, :] = model_rdm_vector
        control_model_rdms_tensor = torch.from_numpy(control_model_rdms_vector)  # (n_control_models, 10296) torch.float32
        ## Regress out control models from target model
        target_model_rdm_tensor_regout = get_tensor_residuals(target_model_rdm_tensor, control_model_rdms_tensor)
        ## Convert to squareform tensor
        model_rdm_tensor_squareform = get_tensor_squareform(target_model_rdm_tensor_regout, 144).to(rsa_device)  # (144, 144) torch.float32
        ## Loop through batches of voxel RDMs
        baseline_corr_list = []
        count_pass_baseline_list = []
        percent_pass_baseline_list = []
        for b_i in range(n_batches):
            brain_rdms_array = batched_brain_rdms_dict[b_i]
            brain_rdms_tensor = torch.from_numpy(brain_rdms_array)  # (n_voxels_per_batch, 10296) torch.float32
            ### Regress out control models from brain RDMs
            brain_rdms_tensor_regout = get_tensor_residuals(brain_rdms_tensor, control_model_rdms_tensor)  # (n_voxels_per_batch, 10296) torch.float32
            ### Conver to squareform tensor
            brain_rdms_tensor_squareform = get_tensor_squareform(brain_rdms_tensor_regout, 144).to(rsa_device)  # (n_voxels_per_batch, 144, 144) torch.float32
            ### Compute correlation (uncorrected)
            baseline_corr_tensor = compute_r(brain_rdms_tensor_squareform, model_rdm_tensor_squareform, triu_ind).cpu()  # (n_voxels_per_batch,) torch.float32
            baseline_corr_list.append(baseline_corr_tensor.numpy())  # (n_voxels_per_batch) list of np.float32 arrays
            with open(log_file, 'a') as f:
                f.write(f'{datetime.now()}, Batch {b_i} baseline correlation done:\n')
                f.write(f'min = {baseline_corr_tensor.min().item()}, max = {baseline_corr_tensor.max().item()}\n')
                f.write(f'mean = {baseline_corr_tensor.mean().item()}, std = {baseline_corr_tensor.std().item()}\n')
                f.write(f'median = {baseline_corr_tensor.median().item()}\n')
            ### Permutation test
            count_pass_baseline_tensor = torch.zeros_like(baseline_corr_tensor, dtype=torch.int8)  # (n_voxels_per_batch,) torch.int8
            for p_i in range(n_permutations):
                perm_ind = permutation_indices[p_i, :]
                if p_i % int(n_permutations/100) == 0:
                    with open(log_file, 'a') as f:
                        f.write(f'{datetime.now()}, Permutation {p_i}\n')
                perm_corr_tensor = compute_r(brain_rdms_tensor_squareform, model_rdm_tensor_squareform, triu_ind, perm_ind).cpu()
                ### Compare with baseline (num_voxels,) torch.float32
                add_count = perm_corr_tensor > baseline_corr_tensor
                add_count = add_count.type(torch.int8)
                count_pass_baseline_tensor += add_count
            percent_pass_baseline_tensor = count_pass_baseline_tensor / n_permutations  # (n_voxels_per_batch,) torch.float32
            count_pass_baseline_list.append(count_pass_baseline_tensor.numpy())  # (n_voxels_per_batch) list of np.int8 arrays
            percent_pass_baseline_list.append(percent_pass_baseline_tensor.numpy())  # (n_voxels_per_batch) list of np.float32 arrays
            with open(log_file, 'a') as f:
                f.write(f'{datetime.now()}, Batch {b_i} {n_permutations} permutations done.\n')
        ## Concatenate batch results
        baseline_corr = np.concatenate(baseline_corr_list, axis=0)  # (n_voxels,) np.float32 array
        count_pass_baseline = np.concatenate(count_pass_baseline_list, axis=0)  # (n_voxels,) np.int8 array
        percent_pass_baseline = np.concatenate(percent_pass_baseline_list, axis=0)  # (n_voxels,) np.float32 array
        ## Fisher transform
        uncorr_z_vector = np.arctanh(baseline_corr)  # (n_voxels,) np.float32 array
        uncorr_z_vector = (uncorr_z_vector - np.nanmean(uncorr_z_vector)) / np.nanstd(uncorr_z_vector)
        ## Plot on brain
        uncorrected_rmap_array = np.zeros_like(brain_mask_array, dtype=np.float32)  # brain_shape np.float32 array
        uncorrected_zmap_array = np.zeros_like(brain_mask_array, dtype=np.float32)  # brain_shape np.float32 array
        uncorrected_pmap_array = np.zeros_like(brain_mask_array, dtype=np.float32)  # brain_shape np.float32 array
        for voxel_i, voxel_info in voxel_indexing_dict.items():
            voxel_x, voxel_y, voxel_z = voxel_info['xyz']
            uncorrected_rmap_array[voxel_x, voxel_y, voxel_z] = baseline_corr[voxel_i]
            uncorrected_zmap_array[voxel_x, voxel_y, voxel_z] = uncorr_z_vector[voxel_i]
            uncorrected_pmap_array[voxel_x, voxel_y, voxel_z] = percent_pass_baseline[voxel_i]
        uncorrected_rmap_img = image.new_img_like(brain_mask_image, uncorrected_rmap_array)  # brain_shape image
        uncorrected_zmap_img = image.new_img_like(brain_mask_image, uncorrected_zmap_array)  # brain_shape image
        uncorrected_pmap_img = image.new_img_like(brain_mask_image, uncorrected_pmap_array)  # brain_shape image
        uncorrected_rmap_img.to_filename(os.path.join(output_folder, f'{analysis_prefix}_partial-rsa_{model_set_name}_uncorrected-rmap.nii.gz'))
        uncorrected_zmap_img.to_filename(os.path.join(output_folder, f'{analysis_prefix}_partial-rsa_{model_set_name}_uncorrected-zmap.nii.gz'))
        uncorrected_pmap_img.to_filename(os.path.join(output_folder, f'{analysis_prefix}_partial-rsa_{model_set_name}_uncorrected-pmap.nii.gz'))
        ## FDR control
        for fdr_alpha in fdr_alpha_list:
            ### FDR controlled zmap
            rejected, q_vector = fdrcorrection(percent_pass_baseline, alpha=fdr_alpha, method='indep', is_sorted=False)  # (n_voxels,) float np.array
            for q_threshold in q_threshold_list:
                fdr_controlled_q_vector = q_vector <= q_threshold  # (n_voxels,) bool np.array
                fdr_controlled_z_vector = uncorr_z_vector * fdr_controlled_q_vector.astype(np.int8)  # (n_voxels,) np.float32 array
                ### Normalize
                # fdr_controlled_z_vector = (fdr_controlled_z_vector - np.nanmean(fdr_controlled_z_vector)) / np.nanstd(fdr_controlled_z_vector)
                ### Construct brain array
                fdr_controlled_zmap_array = np.zeros_like(brain_mask_array, dtype=np.float32)  # brain_shape np.float32 array
                for voxel_i, voxel_info in voxel_indexing_dict.items():
                    voxel_x, voxel_y, voxel_z = voxel_info['xyz']
                    fdr_controlled_zmap_array[voxel_x, voxel_y, voxel_z] = fdr_controlled_z_vector[voxel_i]
                fdr_controlled_zmap_img = image.new_img_like(brain_mask_image, fdr_controlled_zmap_array)  # brain_shape image
                fdr_controlled_zmap_img.to_filename(os.path.join(output_folder, f'{analysis_prefix}_partial-rsa_{model_set_name}_fdr-{fdr_alpha}_q-{q_threshold}.nii.gz'))
                ### Cluster correction: apply 95% threshold
                for z_percentile in z_percentile_list:
                    z_thr_array = fdr_controlled_zmap_array.copy()
                    if z_percentile == 95:
                        z_threshold = 1.96
                    elif z_percentile == 99:
                        z_threshold = 2.58
                    else:
                        two_sided_percentile = 50 + z_percentile/2
                        z_threshold = np.percentile(fdr_controlled_z_vector, two_sided_percentile)
                    z_thr_array[np.abs(z_thr_array) < z_threshold] = 0
                    #### Form clusters
                    all_clusters, n_clusters = label(z_thr_array, cluster_forming_struct)
                    n_zeros = np.count_nonzero(all_clusters == 0)
                    cluster_sizes = np.zeros(n_clusters, dtype=int)
                    for i in range(n_clusters):
                        cluster_sizes[i] = np.count_nonzero(all_clusters == i+1)
                    #### Cluster thresholding
                    for cluster_size_threshold in cluster_size_threshold_list:
                        cluster_corrected_zmap_array = z_thr_array.copy()
                        num_left_clusters = 0
                        for i in range(n_clusters):
                            if cluster_sizes[i] < cluster_size_threshold:
                                cluster_corrected_zmap_array[all_clusters == i+1] = 0
                        #### Save
                        cluster_corrected_zmap_img = image.new_img_like(brain_mask_image, cluster_corrected_zmap_array)  # brain_shape image
                        cluster_corrected_zmap_img.to_filename(os.path.join(output_folder, f'{analysis_prefix}_partial-rsa_{model_set_name}_fdr-{fdr_alpha}_q-{q_threshold}_z-{z_percentile}_cluster-size-{cluster_size_threshold}.nii.gz'))
    with open(log_file, 'a') as f:
        f.write(f'{datetime.now()} Done.\n\n')
