## Dynamic-Neural-Decoding-Pipeline

Figure 1

### Contents
- [Installation](#installation)
- [Usage](#usage)
- [Demos](#demos)
- [TODO List](#todo-list)
- Citation

### Installation
- #### From source
  - **CPU only**
  - **GPU CUDA**
- #### Using a Docker container
  - **CPU only**
  - **GPU CUDA**
- #### Testing installation
  - See [Demos](#demos)

### Usage
#### One-line command
Configure `dndp_job.py` and run 
```
python dndp_job.py
```
#### General procedure
Figure 2
1. Import modules `from utils import *`
2. Initialize a pipeline `dndp(list_of_analyses=[])`. See [analysis module](AnalysisModule.md) for details.
3. Configure the pipeline `dndp.configure(update_param={})`
4. Execute `dndp.run(monitor=True)`

#### Demos
- **Stimulus feature pattern computation**
```
python demo_stimulus_feature_pattern.py
```
- **Representational similarity analysis**
```
python demo_rsa.py
```
- **Multivoxel pattern classification**
```
python demo_mvpa_cls.py
```
- **Representation-similarity-based encoding**
```
python demo_rs_encoding.py
```

### TODO List
- [ ] Support GPU MacOS
- [ ] Support stimulus-feature-based encoding
- [ ] Citation