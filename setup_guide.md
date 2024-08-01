### Set up environment on Mac OS
```
# Install Python 3.11
## On Ubuntu
sudo apt install python3.11
## On Mac OS
brew install python@3.11
# You may need to update $PATH
# export PATH="$PATH:/path/to/python3.11/bin"

# Create a virtual environment named dndp_env
python3.11 -m venv /path/to/dndp_env
# Activate the environment
source /path/to/dndp_env/bin/activate
# Check the location of python interpreter
which python

# Install requirements
python3.11 -m pip install -U -r /path/to/requirements.txt
```

### Tested requirements version
nilearn==0.10.4
scikit-learn==1.5.1
scikit-image==0.24.0
opencv-python==4.10.0.84
pandas==2.2.2
scipy==1.14.0
statsmodels==0.14.2
fastdist==1.1.6
pykeops==2.2.3
torch==2.4.0
torchvision==0.19.0