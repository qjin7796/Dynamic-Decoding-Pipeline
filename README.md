## Dynamic-Neural-Decoding-Pipeline

Figure 1

### Contents
- [Installation](#installation)
- [Usage](#usage)
- [Demos](#demos)
- [TODO List](#todo-list)
- Citation

### Installation
- #### From source (prerequisite: CUDA 11.3)
```
# Clone repository
git clone https://github.com/qjin7796/Dynamic-Neural-Decoding-Pipeline.git
cd Dynamic-Neural-Decoding-Pipeline

# Create a virtual environment
conda create -n dndp_env python=3.10
conda activate dndp_env

# Install dependencies
# See requirements.txt for details
chmod +x setup.sh
./setup.sh
```
- #### Using Docker (prerequisite: docker, nvidia-docker2)
```
# Download latest docker image
docker pull dndp/dndp:latest
```

### Usage
#### One-line command
Configure your `dndp_job.py` and run
```
# Locally
python dndp_job.py

# On a cluster
sbatch dndp_job.slurm

# Using docker
ADD /path/to/dndp_job.py /path/to/docker/image
docker run --runtime=nvidia --rm image_id dndp_job.py
```

#### Interactive mode
1. Import modules `from utils import *`
2. Initialize a pipeline `dndp(list_of_analyses=[])`. See [analysis module](AnalysisModule.md) for details.
3. Configure the pipeline `dndp.configure(update_param={})`
4. Execute `dndp.run(monitor=True)`
5. Check output `print(dndp.summary)`

### Demos
- **Stimulus feature pattern computation**
```
python demos/demo_stimulus_feature_pattern.py
```
- **Representational similarity analysis**
```
python demos/demo_rsa.py
```

### TODO List
- [ ] Support GPU MacOS
- [ ] Support stimulus-feature-based encoding
- [ ] Citation