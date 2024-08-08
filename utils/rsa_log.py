import numpy as np
import os, pickle, json


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
#     (10000, 144) np.uint8 array
# (5) 'brain_mask_filename': str,
#     (x, y, z) uint8 NIfTI image.
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



################### Standard RSA ###################
for search_type in [
    # 'human-20-subjects_glm-raw_searchlight-4mm',
    # 'human-20-subjects_glm-raw_searchlight-5mm',
    # 'human-20-subjects_glm-raw_searchlight-6mm',
    'human-20-subjects_glm-raw_searchlight-7mm',
    'human-20-subjects_glm-raw_searchlight-8mm',
]:
    analysis_1_params = {
        'data_folder': '/data/leuven/347/vsc34741/code/rsa_human',
        'analysis_name': f'batch-1_{search_type}_standard-RSA',
        'group_searchlight_rdms_filename': f'PAO_{search_type}_group-rdms.pickle',
        'rsa_model_sets': {
            'standard': [
                'action-type', 'viewpoint', 'handedness', 'object-type',
                'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                'hand-shape', 'arm-shape', 
            ],
        },
        'n_permutations': 10000,
        'n_batches': 1,
        'categorical_model_rdms_filename': 'PAO_categorical-model-rdms.npz',
        'real_valued_model_rdms_filename': 'PAO_real-valued-model-rdms.npz',
        'permutation_indices_filename': 'PAO_permutation-10k_tuple-list.npz',
        'brain_mask_filename': 'PAO_human-20-subjects_refined-brain-mask.nii.gz',
        'voxel_indexing_dict_filename': 'PAO_human-20-subjects_refined-voxel-dict.npz',
        'fdr_alpha_list': [0.001],
        'q_threshold_list': [0.05, 0.01],
        'smooth_fwhm_list': [3.0, 5.0, 8.0],
    }
    # Save analysis_params to a json file.
    with open(f'batch-1_{search_type}_standard-RSA.json', 'w') as f:
        json.dump(analysis_1_params, f, indent=4)
    
    analysis_2_params = {
        'data_folder': '/data/leuven/347/vsc34741/code/rsa_human',
        'analysis_name': f'batch-3_{search_type}_standard-RSA',
        'group_searchlight_rdms_filename': f'PAO_{search_type}_group-rdms.pickle',
        'rsa_model_sets': {
            'standard': [
                'action-type', 'viewpoint', 'handedness', 'object-type',
                'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                'hand-shape', 'arm-shape', 
            ],
        },
        'n_permutations': 10000,
        'n_batches': 3,
        'categorical_model_rdms_filename': 'PAO_categorical-model-rdms.npz',
        'real_valued_model_rdms_filename': 'PAO_real-valued-model-rdms.npz',
        'permutation_indices_filename': 'PAO_permutation-10k_tuple-list.npz',
        'brain_mask_filename': 'PAO_human-20-subjects_refined-brain-mask.nii.gz',
        'voxel_indexing_dict_filename': 'PAO_human-20-subjects_refined-voxel-dict.npz',
        'fdr_alpha_list': [0.001],
        'q_threshold_list': [0.05, 0.01],
        'smooth_fwhm_list': [3.0, 5.0, 8.0],
    }
    # Save analysis_params to a json file.
    with open(f'batch-3_{search_type}_standard-RSA.json', 'w') as f:
        json.dump(analysis_2_params, f, indent=4)

    analysis_3_params = {
        'data_folder': '/data/leuven/347/vsc34741/code/rsa_human',
        'analysis_name': f'batch-1_{search_type}_partial-RSA',
        'group_searchlight_rdms_filename': f'PAO_{search_type}_group-rdms.pickle',
        'rsa_model_sets': {
            'partial': {
                'action-type-global': [
                    'action-type', 
                    'viewpoint', 'handedness', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                ],
                'action-type-global-hand-shape': [
                    'action-type', 
                    'viewpoint', 'handedness', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                    'hand-shape', 
                ],
                'action-type-global-arm-shape': [
                    'action-type', 
                    'viewpoint', 'handedness', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                    'arm-shape', 
                ],
                'viewpoint-global': [
                    'viewpoint', 
                    'action-type', 'handedness', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                ],
                'viewpoint-global-hand-shape': [
                    'viewpoint', 
                    'action-type', 'handedness', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                    'hand-shape', 
                ],
                'viewpoint-global-arm-shape': [
                    'viewpoint', 
                    'action-type', 'handedness', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                    'arm-shape', 
                ],
                'handedness-global': [
                    'handedness', 
                    'action-type', 'viewpoint', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                ],
                'handedness-global-hand-shape': [
                    'handedness', 
                    'action-type', 'viewpoint', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                    'hand-shape', 
                ],
                'handedness-global-arm-shape': [
                    'handedness', 
                    'action-type', 'viewpoint', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                    'arm-shape', 
                ],
            },
        },
        'n_permutations': 10000,
        'n_batches': 1,
        'categorical_model_rdms_filename': 'PAO_categorical-model-rdms.npz',
        'real_valued_model_rdms_filename': 'PAO_real-valued-model-rdms.npz',
        'permutation_indices_filename': 'PAO_permutation-10k_tuple-list.npz',
        'brain_mask_filename': 'PAO_human-20-subjects_refined-brain-mask.nii.gz',
        'voxel_indexing_dict_filename': 'PAO_human-20-subjects_refined-voxel-dict.npz',
        'fdr_alpha_list': [0.001],
        'q_threshold_list': [0.05, 0.01],
        'smooth_fwhm_list': [3.0, 5.0, 8.0],
    }
    # Save analysis_params to a json file.
    with open(f'batch-1_{search_type}_partial-RSA.json', 'w') as f:
        json.dump(analysis_3_params, f, indent=4)

    analysis_4_params = {
        'data_folder': '/data/leuven/347/vsc34741/code/rsa_human',
        'analysis_name': f'batch-3_{search_type}_partial-RSA',
        'group_searchlight_rdms_filename': f'PAO_{search_type}_group-rdms.pickle',
        'rsa_model_sets': {
            'partial': {
                'action-type-global': [
                    'action-type', 
                    'viewpoint', 'handedness', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                ],
                'action-type-global-hand-shape': [
                    'action-type', 
                    'viewpoint', 'handedness', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                    'hand-shape', 
                ],
                'action-type-global-arm-shape': [
                    'action-type', 
                    'viewpoint', 'handedness', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                    'arm-shape', 
                ],
                'viewpoint-global': [
                    'viewpoint', 
                    'action-type', 'handedness', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                ],
                'viewpoint-global-hand-shape': [
                    'viewpoint', 
                    'action-type', 'handedness', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                    'hand-shape', 
                ],
                'viewpoint-global-arm-shape': [
                    'viewpoint', 
                    'action-type', 'handedness', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                    'arm-shape', 
                ],
                'handedness-global': [
                    'handedness', 
                    'action-type', 'viewpoint', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                ],
                'handedness-global-hand-shape': [
                    'handedness', 
                    'action-type', 'viewpoint', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                    'hand-shape', 
                ],
                'handedness-global-arm-shape': [
                    'handedness', 
                    'action-type', 'viewpoint', 'object-type', 
                    'global-luminance', 'global-Gabor-mean-response', 'global-flow',
                    'arm-shape', 
                ],
            },
        },
        'n_permutations': 10000,
        'n_batches': 3,
        'categorical_model_rdms_filename': 'PAO_categorical-model-rdms.npz',
        'real_valued_model_rdms_filename': 'PAO_real-valued-model-rdms.npz',
        'permutation_indices_filename': 'PAO_permutation-10k_tuple-list.npz',
        'brain_mask_filename': 'PAO_human-20-subjects_refined-brain-mask.nii.gz',
        'voxel_indexing_dict_filename': 'PAO_human-20-subjects_refined-voxel-dict.npz',
        'fdr_alpha_list': [0.001],
        'q_threshold_list': [0.05, 0.01],
        'smooth_fwhm_list': [3.0, 5.0, 8.0],
    }
    # Save analysis_params to a json file.
    with open(f'batch-3_{search_type}_partial-RSA.json', 'w') as f:
        json.dump(analysis_4_params, f, indent=4)

