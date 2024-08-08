### Implemented from: https://nilearn.github.io/dev/auto_examples/07_advanced/plot_beta_series.html
### Apart from estimating beta-series per event per subject per run, a contrast test is also done per event per subject across runs for comparison with Prosper monkey data RSA.
### Output 1: one beta map per subject per run per event (every run has 72 events).
### Output 2: one zmap per subject per event (every event occurs in 4 or 5 runs).
### Note: https://neurostars.org/t/multiple-sessions-not-in-all-sessions-an-onset-for-certain-conditions/3090
### Since LSS-GLM approach simply uses single trial's beta map, there's no reason to include other runs even if only half of 144 conditions appear in one run.
### Including additional runs won't impact the betas for the single-trial condition, unless the other session is modeled as if it was the same session, 
###     which introduces bias to the model estimation and should be avoided.


import os, pickle, datetime, numpy as np, pandas as pd
from nilearn import plotting, image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn._utils.niimg_conversions import check_niimg
from scipy.stats import gamma


def mion_response_function(tr, oversampling=16, onset=0.0, time_length=70):
    # oversampling (temporal oversampling factor for calculating regressor) = T or fMRI_T in spm_hrf
    # time_length (hrf kernel length in seconds) = p(7) in spm_hrf
    dt = tr / oversampling
    time_stamps = np.linspace(
        0, time_length, np.rint(float(time_length) / dt).astype(int)
    )
    time_stamps -= onset
    # define peak gamma function
    delay = 1.55 
    dispersion = 5.5
    peak_gamma = gamma.pdf(time_stamps, delay, loc=0, scale=dispersion)
    hrf = peak_gamma / peak_gamma.sum()
    hrf *= -1
    return hrf


def lss_transformer(df, row_number):
    df = df.copy()
    # Determine trial name and the other trials
    trial_name = df.loc[row_number, "trial_type"]
    trial_type_series = df["trial_type"]
    trial_type_series = trial_type_series.loc[
        trial_type_series != trial_name
    ]
    trial_type_list = trial_type_series.index.tolist()
    # Keep the name of the isolated trial and collapse all the other into "movie"
    for rowI in trial_type_list:
        df.loc[rowI, "trial_type"] = "movie"
    # print('lss_transformer transformed', len(trial_type_list), 'trials to movie and isolated', trial_name)
    return df, trial_name


## Define parameters
data_folder = "/scratch/leuven/347/vsc34741/code/rsa_preprocessing/bids_monkey_D99"
space_name = "D99"
group_mean_mask = "/data/leuven/347/vsc34741/code/rsa_preprocessing/D99_template_1mm_mask_lr.nii"
group_mean_mask_img = image.load_img(group_mean_mask)
group_mean_mask_data = image.get_data(group_mean_mask_img)
subject_list = ['sub-Radja']
output_folder = "/scratch/leuven/347/vsc34741/code/rsa_preprocessing/lss-glm_D99-raw-spm1-hmc-cosine_sub-Radja"
os.makedirs(output_folder, exist_ok=True)
events_folder = "/data/leuven/347/vsc34741/code/rsa_preprocessing/events_monkey"
param_slice_time_ref = 0.1  # time of the reference slice used in slice timing correction, fMRIPrep uses the median slice
param_hrf_model = mion_response_function  # use spm12 hrf model for mion, default is glover hrf for bold
hrf_model_str = "mion"
confounds_str = "6-motion-confounds"
param_drift_model = "cosine"  # drift model, default is cosine
param_drift_order = 3  # drift model order, default is 1
param_smoothing_fwhm = None  # smoothing kernel size
smoothing_str = "raw"

condition_labels = np.load("/data/leuven/347/vsc34741/code/rsa_preprocessing/condition_labels.npy")  
# list of 144 conditions that were split into two runs during the experiment; output are saved in dict with keys = condition_list
condition_list = condition_labels.tolist()
task_name = "PAO"
param_t_r = 2.0  # repetition time is 1 second
param_signal_scaling = False  # do not apply mean scaling to each voxel again, default is 0 - mean scaling each voxel with respect to time
param_high_pass = 1/128  # high pass filter frequency
# Selected subjects and runs
subject_runs = {
    'sub-Radja': {
        'ses-191008': [1,2,3,4,5,6,8,9,10,11,12,13],
        'ses-191009': [1,2,3,4,5,6,7,8,9,10,12,13,14,15],
        'ses-191010': [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17],
    },
    'sub-Dobby': {
        'ses-190917': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        'ses-190918': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    },
    'sub-Elmo': {
        'ses-190814': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        'ses-190816': [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17],
    }
}

### Fetch subject data
all_subject_data = {
    'subject_list': subject_list,
    'subject_n_sessions': [],
    'subject_n_runs': [],
    'data_folder': data_folder,
    'output_folder': output_folder,
    'run_list': subject_runs,
}
for subject in subject_list:
    subject_data_path = os.path.join(data_folder, subject)
    all_subject_data[subject] = {}
    subject_run_list = []
    subject_run_imgs = []
    subject_design_matrices = []
    ## Load subject preprocessed functional data
    subject_n_sessions = len(subject_runs[subject])
    print('- Checking', subject, datetime.datetime.now())
    print('- Subject has', subject_n_sessions, 'sessions.')
    all_subject_data['subject_n_sessions'].append(subject_n_sessions)
    subject_n_runs = 0
    for sub_ses, sub_ses_run_list in subject_runs[subject].items():
        ## Read functional data
        for runI in sub_ses_run_list:
            run_str = '{:02d}'.format(runI)
            subject_run_list.append(runI)
            subject_n_runs += 1
            # Read path to preproc func image
            path_run_img = os.path.join(
                subject_data_path, 'func',
                f"{subject}_{sub_ses}_task-PAO_run-{run_str}_space-{space_name}_desc-preproc_mion.nii"
            )
            subject_run_imgs.append(path_run_img)
            print(f'--- read {subject} {sub_ses} run-{runI} path: {path_run_img}')
            # Read events table
            path_run_events = os.path.join(
                events_folder, 
                f"{subject}_{sub_ses}_task-PAO_run-{run_str}_events.tsv"
            )
            subject_run_events = pd.read_csv(path_run_events, sep='\t')
            # Read confounds
            if confounds_str == '5-high-variance-confounds':
                subject_run_confounds = pd.DataFrame(image.high_variance_confounds(path_run_img, n_confounds=5))
            elif confounds_str == '6-motion-confounds':
                path_run_confounds = os.path.join(
                    subject_data_path, 'func',
                    f"{subject}_{sub_ses}_task-PAO_run-{run_str}_desc-confounds_timeseries.tsv"
                )
                subject_run_confounds_table = pd.read_csv(path_run_confounds, sep='\t')
                subject_run_confounds = subject_run_confounds_table[
                    ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
                ]
            else:
                raise ValueError('confounds_str not recognized.')
            # Make LSS-GLM design matrices:
            # (1) Transform events table to LSS models: for each trial, make one table in which the trial name is 
            #     kept as it is and all the other trials collapsed into a single condition "movie"
            # (2) Append head motion confounds
            run_img = check_niimg(path_run_img, ensure_ndim=4)
            n_scans = image.get_data(run_img).shape[3]
            confounds_matrix = subject_run_confounds.values
            if confounds_matrix.shape[0] != n_scans:
                raise ValueError('Rows in confounds does not match'
                                f'n_scans in run_img at index {runI}.')
            confounds_names = subject_run_confounds.columns.tolist()
            start_time = param_slice_time_ref * param_t_r
            end_time = (n_scans - 1 + param_slice_time_ref) * param_t_r
            frame_times = np.linspace(start_time, end_time, n_scans)
            lss_design_matrices = {cond: [] for cond in subject_run_events['trial_type']}
            for trialI in range(subject_run_events.shape[0]):
                # Design an LSS model for the current trial
                lss_events_df, trial_condition = lss_transformer(subject_run_events, trialI)
                # Make design matrices for all runs
                if param_drift_model == 'cosine' or param_drift_model == None:
                    design_matrix = make_first_level_design_matrix(
                        frame_times, events=lss_events_df, hrf_model=param_hrf_model,
                        drift_model=param_drift_model, high_pass=param_high_pass,
                        add_regs=confounds_matrix, add_reg_names=confounds_names,
                    )
                elif param_drift_model == 'polynomial':
                    design_matrix = make_first_level_design_matrix(
                        frame_times, events=lss_events_df, hrf_model=param_hrf_model,
                        drift_model=param_drift_model, drift_order=param_drift_order,
                        add_regs=confounds_matrix, add_reg_names=confounds_names,
                    )
                else:
                    raise ValueError('param_drift_model not recognized.')
                lss_design_matrices[trial_condition] = design_matrix
            subject_design_matrices.append(lss_design_matrices)
    all_subject_data[subject]['run_imgs'] = subject_run_imgs
    all_subject_data[subject]['lss_glm_design_matrices'] = subject_design_matrices
    print('- Subject has', subject_n_runs, 'runs.')
    all_subject_data[subject]['n_runs'] = subject_n_runs

### Save
all_subject_data_save_path = os.path.join(output_folder, f"fetched_data_for_LSS_GLM.sav")
pickle.dump(all_subject_data, open(all_subject_data_save_path, 'wb')) # loaded_model = pickle.load(open(filename, 'rb')) or pd.read_pickle(filename)

### Create and fit a first level model for each subject individually
print(datetime.datetime.now(), 'LSS-GLM')
for subject in subject_list:
    print(f"- Processing {subject}, {datetime.datetime.now()}")
    sub_out_dir = os.path.join(output_folder, subject)
    os.makedirs(sub_out_dir, exist_ok=True)
    subject_run_imgs = all_subject_data[subject]['run_imgs']
    subject_design_matrices = all_subject_data[subject]['lss_glm_design_matrices']
    subject_lss_glm_zmaps = {} # one ffx map per condition across runs, saved as a 4D array with the last dimension the 144 conditions
    subject_lss_glm_tmaps = {} # one ffx map per condition across runs, saved as a 4D array with the last dimension the 144 conditions
    subject_lss_glm_bmaps = {} # one ffx map per condition across runs, saved as a 4D array with the last dimension the 144 conditions
    subject_lss_glm_beta_series = {cond: [] for cond in condition_list}  # one map per condition per run, each condition has its independent model
    subject_lss_glm_ffx_output = {}
    for condition_i, condition_name in enumerate(condition_list):
        print(f"--- condition {condition_i} {condition_name}, {datetime.datetime.now()}")
        subject_condition_cmaps = []
        subject_condition_vmaps = []
        for runI, run_img in enumerate(subject_run_imgs):
            # print('----- run', runI+1, 'condition name', condition_name)
            lss_design_matrix = subject_design_matrices[runI][condition_name]
            # (1) Fit subject-level condition- and run-independent lss_glm
            lss_glm = FirstLevelModel(
                mask_img=group_mean_mask,
                smoothing_fwhm=param_smoothing_fwhm,
                signal_scaling=param_signal_scaling,
            )
            lss_glm = lss_glm.fit(run_img, design_matrices=lss_design_matrix)
            subject_lss_glm_stats = lss_glm.compute_contrast(f'{condition_name}_mion_response_function', output_type='all')
            subject_condition_cmaps.append(subject_lss_glm_stats['effect_size'])
            subject_condition_vmaps.append(subject_lss_glm_stats['effect_variance'])
            subject_lss_glm_beta_series[condition_name].append(subject_lss_glm_stats['effect_size'])
            # (2) Plot design matrix
            sub_img_out_dir = os.path.join(sub_out_dir, 'design_matrices')
            if not os.path.exists(sub_img_out_dir):
                os.makedirs(sub_img_out_dir)
            design_matrix_img = plotting.plot_design_matrix(
                lss_design_matrix, output_file=os.path.join(
                    sub_img_out_dir, 
                    f"LSS-GLM_{subject}_run-{runI}_{condition_name}_design-matrix.png"
                )
            )
            contrast_matrix_img = plotting.plot_contrast_matrix(
                f'{condition_name}_mion_response_function', 
                design_matrix=lss_design_matrix,
                output_file=os.path.join(
                    sub_img_out_dir, 
                    f"LSS-GLM_{subject}_run-{runI}_{condition_name}_contrast-matrix.png"
                )
            )
        ## Estimate subject FFX of the specific condition across runs
        subject_condition_ffx_cmap, subject_condition_ffx_vmap, subject_condition_ffx_tmap, subject_condition_ffx_zmap = compute_fixed_effects(
            subject_condition_cmaps, subject_condition_vmaps, mask=group_mean_mask, return_z_score=True
        )
        subject_lss_glm_zmaps[condition_name] = subject_condition_ffx_zmap
        subject_lss_glm_tmaps[condition_name] = subject_condition_ffx_tmap
        subject_lss_glm_bmaps[condition_name] = subject_condition_ffx_cmap
        subject_lss_glm_ffx_output[condition_name] = {
            'effect_size': subject_condition_cmaps,
            'effect_variance': subject_condition_ffx_vmap,
            'stat': subject_condition_ffx_tmap,
            'z_score': subject_condition_ffx_zmap,
        } # save all outputs
        ## Save zmap
        sub_trial_zmap_out_dir = os.path.join(sub_out_dir, 'zmaps')
        if not os.path.exists(sub_trial_zmap_out_dir):
            os.makedirs(sub_trial_zmap_out_dir)
        subject_lss_glm_zmap_tosave_path = os.path.join(
            sub_trial_zmap_out_dir, 
            f"LSS-GLM_{subject}_{condition_name}_zmap.nii.gz"
        )
        subject_condition_ffx_zmap.to_filename(subject_lss_glm_zmap_tosave_path)
        ## Save tmap
        sub_trial_tmap_out_dir = os.path.join(sub_out_dir, 'tmaps')
        if not os.path.exists(sub_trial_tmap_out_dir):
            os.makedirs(sub_trial_tmap_out_dir)
        subject_lss_glm_tmap_tosave_path = os.path.join(
            sub_trial_tmap_out_dir, 
            f"LSS-GLM_{subject}_{condition_name}_tmap.nii.gz"
        )
        subject_condition_ffx_tmap.to_filename(subject_lss_glm_tmap_tosave_path)
        ## Save bmap
        sub_trial_bmap_out_dir = os.path.join(sub_out_dir, 'bmaps')
        if not os.path.exists(sub_trial_bmap_out_dir):
            os.makedirs(sub_trial_bmap_out_dir)
        subject_lss_glm_bmap_tosave_path = os.path.join(
            sub_trial_bmap_out_dir, 
            f"LSS-GLM_{subject}_{condition_name}_bmap.nii.gz"
        )
        subject_condition_ffx_cmap.to_filename(subject_lss_glm_bmap_tosave_path)
    ## Concat FFX zmaps into a 4D NIfTI image array (last dimension = conditions) and save
    subject_lss_glm_zmap_concat = image.concat_imgs([zmap for cond, zmap in subject_lss_glm_zmaps.items()])
    subject_lss_glm_zmap_concat.to_filename(os.path.join(
        sub_out_dir, 
        f"LSS-GLM_{subject}_{smoothing_str}_{confounds_str}_drift-{param_drift_model}_hrf-{hrf_model_str}_hpf-128_concat-zmaps.nii.gz"
    ))
    ## Compute FFX zmaps mean and variance across conditions and save
    ## subject_lss_glm_zmean: mean zmap across conditions
    subject_lss_glm_zmean = image.mean_img(subject_lss_glm_zmap_concat)
    subject_lss_glm_zmean.to_filename(os.path.join(
        sub_out_dir,
        f"LSS-GLM_{subject}_{smoothing_str}_{confounds_str}_drift-{param_drift_model}_hrf-{hrf_model_str}_hpf-128_zmean.nii.gz"
    ))
    subject_lss_glm_zvar = image.math_img("np.std(img, axis=3)", img=subject_lss_glm_zmap_concat)
    subject_lss_glm_zvar.to_filename(os.path.join(
        sub_out_dir,
        f"LSS-GLM_{subject}_{smoothing_str}_{confounds_str}_drift-{param_drift_model}_hrf-{hrf_model_str}_hpf-128_zvar.nii.gz"
    ))
    ## Plot
    plotting.plot_stat_map(
        subject_lss_glm_zmean, threshold=None,
        colorbar=True, display_mode='z', cut_coords=20,
        title=f"{subject} LSS-GLM zmean (thr None)",
        output_file=os.path.join(
            sub_out_dir,
            f"LSS-GLM_{subject}_{smoothing_str}_{confounds_str}_drift-{param_drift_model}_hrf-{hrf_model_str}_hpf-128_zmean.png"
        )
    )
    plotting.plot_stat_map(
        subject_lss_glm_zvar, threshold=None,
        colorbar=True, display_mode='z', cut_coords=20,
        title=f"{subject} LSS-GLM zvar (thr None)",
        output_file=os.path.join(
            sub_out_dir,
            f"LSS-GLM_{subject}_{smoothing_str}_{confounds_str}_drift-{param_drift_model}_hrf-{hrf_model_str}_hpf-128_zvar.png"
        )
    )
print(datetime.datetime.now(), 'Done.')
