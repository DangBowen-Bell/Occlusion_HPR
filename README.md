# Reconstructing 3D Human Pose from RGB-D Data with Occlusions (PG 2023)

[[Project Page](https://dangbowen-bell.github.io/projects/pg2023/)]

![Teaser](images/teaser.png)

Code repository for the paper "Reconstructing 3D Human Pose from RGB-D Data with Occlusions".


## Contents

1. [Code Structure](#cocde-structure)
2. [Install](#install)
3. [Data](#data)
4. [Run](#run)
5. [Reference](#reference)


## Code Structure

`fznet`: Free zone network (independent of other code).

`hsiviewer`: Human interaction dataset viewer. Refer to [hsiviewer/README.md](hsiviewer/README.md) for usage instructions.

`cfg_files`: Parameters for optimization.

`misc`: Miscellaneous parameters, functions, or classes.

```
|--constants.py   # Common parameters such as paths.
|--data_io.py     # Class for loading the prox dataset easily.
|--utils.py       # Utility functions.
|--fps.py         # Farthest point sample (FPS) in PyTorch implementation.
|--model.py       # Same as the model.py in fznet.
```

`prox`: Optimization.

`make_data.py`: Scripts for making data used to train the free zone network.

`run.py`: Scripts for running the optimization.


## Install

We train the free zone network on a server without screen. To visualize the optimization process, we run optimization on another computer. 

#### Free Zone Network

You can use [envs/fznet.txt](envs/fznet.txt) to create the virtual environment.

The free zone network part requires the Chamfer3D package. Please refer to [this](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) for the installation of the package.

#### Optimization

You can use [envs/opt.txt](envs/opt.txt) to create the virtual environment.

The optimization part is based on [PROX](https://github.com/mohamedhassanmus/prox). Please refer to its code repository for the installation of the environment.


## Data

Before running the code, you need to download the following data:

- [PROX](https://prox.is.tue.mpg.de/) dataset.
- [POSA](https://posa.is.tue.mpg.de/) extension for PROX dataset.
- [SMPL-X](https://smpl-x.is.tue.mpg.de/) human body model.

There are also other optional data:

- [LEMO](https://sanweiliti.github.io/LEMO/LEMO.html) fitting for PROX dataset.
- [PROX-E](https://github.com/yz-cnsdqz/PSI-release) extension for PROX dataset.

We did not use these data in the formal paper, and you can run the code without them.


## Run

You need to train the free zone network before running the optimization.

#### Free Zone Network

Make your own data before training:

```
python make_data.py
```

The generated data will appear as follows:

```
FZNet
|--prox
   |--recording1
      |--frame1.npz
      |--...
   |--...
|--prox_info.json           # Per-frame information.
|--prox_split_info.json     # Train/test split information.
```

There will also be a `vertice_pairs.npy` file used for volume matching.

Then you can train the free zone network: 

```
python train.py --name exp_name
```

#### Optimization

After training the free zone network, you will get a trained model at `fznet/logs/exp_name/checkpoints/best_model.pt`. You can use it during the optimization:

```
python run.py
```

By default, this will process all test frames in offscreen mode. You can modify the parameters for debugging or visualizing the optimization process.


## Reference

We referred to some model code of [Grasping Field](https://github.com/korrawe/grasping_field). The optimization code is based on [PROX](https://github.com/mohamedhassanmus/prox). Thanks to these authors for their great work.
