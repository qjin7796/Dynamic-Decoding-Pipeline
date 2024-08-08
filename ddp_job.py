# Pipeline configuration
# Author: Qiuhan Jin
# Date: 2024-08-01
#
# Configuration steps:
# 1. Define a list of analyses to be performed
#    Currently supported analyses: ["mri_feature", "mvpa_cls", "mvpa_cls_pattern", "image_feature", "video_feature", "feature_pattern", "pattern_comparison"]
# 2. Define parameters for each of the analyses defined
#    Parameters include analysis-specific parameters and computing-resource parameters
# 3. Define the input and output paths
#
# Configuration command:
# One-line mode: edit ddp_job.py directly.
# Interactive mode: 
#   from utils import *
#   my_ddp = ddp(<list_of_analysis_names>)
#   my_ddp.update(<dict_of_params>)
#
# Execution command:
# One-line mode: python ddp_job.py
# Interactive mode: my_ddp.run()


# 1. List of analyses to be performed
# Example: representational similarity analysis (rsa) of MRI and video data
# analysis_to_do = ["mri_feature", "video_feature", "feature_pattern", "pattern_comparison"]
# Example: multi-voxel pattern classification (mvpa_cls)
# analysis_to_do = ["mri_feature", "mvpa_cls"]
# Example: multi-voxel pattern classification (mvpa_cls) and comparison with behavior
# analysis_to_do = ["mri_feature", "mvpa_cls", "mvpa_cls_pattern", "pattern_comparison"]

# 2. Parameters for each of the analyses to be performed
analysis_param_dict = {
    "video_feature": {
        "video_list": [],  # list of str or /path/to/video_list.txt
        "input_path": "/path/to/list/of/videos",
        "output_path": "/path/to/output",
        "feature": {
            
        }
    },
    "mri_feature": {
        "input_path": "/path/to/bids_derivatives",  # in bids format
        "output_path": "/path/to/output",
        "glm": {
            "mask": None,  # str, path to mask file or None
            "tr": 0,  # float, repitition time in seconds
            "slice_time_ref": 0,  # float, default=0
            "signal_scaling": 0,  # False, int or (int, int), default=0
            "contrast": None,  # str, contrast variable in events.tsv
            "contrast_type": "lss",  # {"lss", "lsa"}, default="lss"
            "confounds": [],  # list of str, confound variables in confounds.tsv
            "hrf_model": "spm",  # {"spm", "glover", "mion"}, default="spm"
            "time_deriv": False,  # if True, the time derivative regressor is added to the design matrix
            "noise_model": "ar1",  # {"ar1", "ols"}, default="ar1"
            "drift_model": "cosine",  # {"cosine", "polynomial", None}, default="cosine"
            "drift_order": 1,  # int, default=1, only used if drift_model is "polynomial"
            "high_pass": 0.01,  # float, default=0.01, only used if drift_model is "cosine"
            "smoothing_fwhm": 3,  # float, smoothing kernel size in mm, 0 for no smoothing
        },
    },
    "mvpa_cls": {
        "input_path": "/path/to/bids_derivatives",  # in bids format
        "output_path": "/path/to/output",
        "mvpa_cls": {
            "mask": None,  # str, path to ROI mask file, None for whole space
            "searchlight_radius": 0,  # float, searchlight radius in mm, 0 for no searchlight
            "stats": "t-value",  # {"t-value", "z-score", "psc"}, default="t-value"
            "smoothing_fwhm": 3,  # float, smoothing kernel size in mm, 0 for no smoothing
            "estimator": "svc",  # {"svc", "svc_l1", "svc_l2", "logistic", "ridge"}, default="svc"
            "cv": 10,  # int, number of folds for cross-validation, default=10
            "scoring": "roc_auc",  # {"accuracy", "f1", "precision", "recall", "roc_auc"} or Callable, default="roc_auc"
        },
    },
}

