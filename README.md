# PanoFlow
The implementaions of [PanoFlow: Learning Optical Flow for Panoramic Images](https://arxiv.org/pdf/2202.13388.pdf). 
We achieve state-of-the-art accuracy on the public OmniFlowNet dataset and the proposed Flow360 dataset.
This repository is built on the basis of [CSFlow](https://github.com/MasterHow/CSFlow).


![](results/compare.png)

# Flow360 Datset
![](results/Flow360.png)
Any one can download our Flow360 dataset via this link.
```
https://share.weiyun.com/SoXICYgh
```

# Install
```
python setup.py develop
```

# Devs
Hao Shi，YiFan Zhou

# Folder Hierarchy
\* local: you should create this folder in your local repository and these folders will not upload to remote repository.
```
├── data (local)            # Store test/training data
├── opticalflow             # All source code
|   ├─ api                  # Called by tools
|   ├─ core                 # Core code call by other directorys. Provide dataset, models ....
|   |   ├─ dataset          # I/O of each dataset
|   |   ├─ model            # Models, includeing all the modules that derive nn.module
|   |   ├─ util             # Utility functions
├── tools                   # Scripts for test and train
├── work_dirs (local)       # For developers to save thier own codes and assets
```

# The Code is coming soon.