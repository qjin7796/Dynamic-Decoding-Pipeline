import numpy as np, cv2


default_params = {
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
            "t_r": 0,  # float, repitition time in seconds
            "smoothing_fwhm": 3,  # float, smoothing kernel size in mm, 0 for no smoothing
            "estimator": "svc",  # {"svc", "svc_l1", "svc_l2", "logistic", "ridge"}, default="svc"
            "cv": 10,  # int, number of folds for cross-validation, default=10
            "scoring": "roc_auc",  # {"accuracy", "f1", "precision", "recall", "roc_auc"} or Callable, default="roc_auc"
            "standardize": True,  # bool, default=True
            "n_jobs": -1,  # int, number of jobs for parallel processing, default=-1, i.e. all processors
            "random_state": None,  # int, RandomState instance or None, default=None
        },
    },
}


def read_video(video: str) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)