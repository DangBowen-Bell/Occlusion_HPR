# 3D Viewer for Human-Scene Interaction Dataset 

Code repository for a 3D viewer to visualize some Human-Scene Interaction datasets.


## Features

- Visualizes images, scenes, depth scans, cameras, and human bodies together.
- Supports multiple human-scene interaction datasets, such as:
    - [PROX](https://prox.is.tue.mpg.de/)
    - ...


## Install

This code is based on [Open3D](http://www.open3d.org/) and have been tested using version 0.14.1.


## Data

Before running the code, you need to download the following data:

- [SMPL-X](https://smpl-x.is.tue.mpg.de/) human body model.
- Dataset you want to visualize, such as [PROX](https://prox.is.tue.mpg.de/).

After downloading the data, you will need to modify the `misc/constants.py` to ensure that the data paths are correctly configured.


## Run

```
python hsiviewer.py
```

Enjoy using this viewer and feel free to report any issues or bugs you may encounter.

The viewer and dataloader are seperated for better scalability. If you want to visualize other datasets, you can refer to `data_io.py` to learn how to create your own dataloader.


## Reference

I used the [example](https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/vis_gui.py) provided by Open3D as a template to design the 3D viewer. I would like to express my gratitude to the authors of the example for their great work.
