## Dynamic-Neural-Decoding

### Figure

### Contents
- [Installation](#installation)
- [Usage](#usage)
- [Demos](#demos)
- [TODO List](#todo-list)
- [Citation](#citation)

### Installation
- #### From source
  - **CPU only**
  - **GPU CUDA**
- #### Using a Docker container
  - **CPU only**
  - **GPU CUDA**
- #### Testing installation
  - **Testing end-to-end RSA with CPU**
  - **Testing end-to-end RSA with GPU CUDA**
  - **Testing end-to-end decoding with CPU**
  - **Testing end-to-end decoding with GPU CUDA**

### Usage
#### General procedure
1. **Define workflow**
  Function: dnd.define(user_dict)
  User input: a dictionary of modules and functions to use.
    Example: `{brain_activity_pattern: [lss_glm, group_rdm], stimulus_feature_pattern, rsa]`
  Output: dnd.workflow, parameters for every function in the specisfied module; dnd.log, user input.
    Example: `dnd.workflow = {
      brain_activity_pattern: {
          bids_derivative_path: None, # raise error if None
          output_path: None, # if None, create brain_activity_pattern.bids_derivative_path/brain_activity_pattern
          subject_list: [],
          lss_glm: {fwhm: 1.5, drift: cosine, hrf: spm, ...},
          # lsa_glm: {},
          group_rdm: {metric: euclidean, exclude_subject: []},
          # individual_rdm: {},
      },
      stimulus_feature_pattern: {
          stimulus_path: None, # raise error if None
          output_path: None, # if None, create stimulus_feature_pattern.stimulus_path/stimulus_feature_pattern
          stimulus_type: avi, # avi/mp4/npz/pkl/jpg/png
          stimulus_list: [], # list of video/image names
          features: [luminance, contrast, optical-flow, hog, ...], # list of features
          feature_mask_list: [], # a list of masks to filter features
          metric: euclidean,
      },
      rsa: {
          input_path: None, # if None, use brain_activity_pattern.output_path
          output_path: None, # if None, create rsa.input_path/rsa
          target_list: [], # group_rdm/individual_rdm
          metric: cosine,
          correction: {fdr-alpha: 0.05, q-thr: 0.05, z-thr: 1.96, cluster-size: 10},
      },
   }`
3. **Define data and parameters**
  Function: dnd.update(user_dict)
  User input: update dnd.workflow keys and values
    Example: dnd.update({'brain_activity_pattern': 'lss_glm': {'fwhm': 0}}) will change dnd.workflow.brain_activity_pattern.lss_glm.fwhm from 1.5 to 0.
  Output: updated dnd.workflow and dnd.log.
5. **Execute and monitor progress**
  Function: dnd_run(monitor=True)
  User call dnd_run()
7. **Visualize results**
  Function: dnd_vis()

### Demos
- #### Stimulus feature patterns
- #### RSA
- #### MVPA classification

### TODO List
- #### Using GPU MacOS
- #### Stimulus-model-based encoding
- #### Representational-similarity-based encoding

### Citation
