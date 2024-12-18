# Self-Supervised Contrastive Learning for Semantic Segmentation of Outdoor LiDAR Point Clouds

Installing pre-requisites:

`sudo apt install build-essential python3-dev libopenblas-dev`

`pip3 install -r requirements.txt`

`pip3 install torch ninja`

Installing MinkowskiEngine with CUDA support:

`pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps`

# SegContrast with Docker

Inside the `docker/` directory there is a `Dockerfile` to build an image to run SegContrast. You can build the image from scratch or download the image from docker hub by:

```
docker pull nuneslu/segcontrast:minkunet
```

Then start the container with:

```
docker run --gpus all -it --rm -v /PATH/TO/SEGCONTRAST:/home/segcontrast segcontrast /bin/zsh
```

# Data Preparation

Download [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) inside the directory ```./Datasets/SemanticKITTI/datasets```. The directory structure should be:

```
./
└── Datasets/
    └── SemanticKITTI
        └── dataset
          └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	├── 000000.bin
            |   |	├── 000001.bin
            |   |	└── ...
            │   └── labels/ 
            |       ├── 000000.label
            |       ├── 000001.label
            |       └── ...
            ├── 08/ # for validation
            ├── 11/ # 11-21 for testing
            └── 21/
                └── ...
```
Download [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/user_login.php) inside the directory ```./Datasets/KITTI-360```. The directory structure should be:
```
./
└── Datasets/
    └── KITTI360
        └── train
          └── sequences
            ├── 00/           
            │   ├── {start_frame:0>10}_{end_frame:0>10}.ply
                └── ...
        └── validation
            └── sequences
                ├── 00/
                │   ├── {start_frame:0>10}_{end_frame:0>10}.ply
                    └── ...
        └── test
            └── sequences
                ├── 08/
                │   ├── {start_frame:0>10}_{end_frame:0>10}.ply
                |   └── ...
                ├── 18/
                │   ├── {start_frame:0>10}_{end_frame:0>10}.ply
                    └── ...
```

First, we need to prepare large point clouds of KITTI-360 for the input of the network. We follow the instructions of [Mahmoudi Kouhi, Reza et al.](https://www.mdpi.com/2072-4292/15/4/982) to prepare the data:

```
python3 ./data_preparation/fps_knn_threading.py --path ./Datasets/KITTI-360/train \
         --save-path ./Datasets/KITTI-360/fps_knn --split train
python3 ./data_preparation/fps_knn_threading.py --path ./Datasets/KITTI-360/validation \
         --save-path ./Datasets/KITTI-360/fps_knn --split validation
```

Now we need to segment the generated point clouds using RANSAC and DBScan:

```
python3 ./data_utils/segmentation.py --dataset KITTI360 --path ./Datasets/KITTI-360/fps_knn \
         --save-path ./Datasets/segmented_views --split train --seq-ids [0,2,3,4,5,6,7,9,10]
python3 ./data_utils/segmentation.py --dataset SemanticKITTI --path ./Datasets/SemanticKITTI \
         --save-path ./Datasets/segmented_views --split train --seq-ids [0,1,2,3,4,5,6,7,9,10]
```

# Reproducing the results

Run the following to start the pre-training:

```
python3 contrastive_train.py --use-cuda
```

The default parameters, e.g., learning rate, batch size and epochs are already the same as the paper.

After pre-training you can run the downstream fine-tuning with:

```
python3 downstream_train.py --use-cuda --checkpoint segment_contrast --load-checkpoint --batch-size 2 --epochs 15
```

We provide in `tools` the `contrastive_train.sh` and `downstream_train.sh` scripts to reproduce the results pre-training and fine-tuning with the different label percentages shown on the paper:

For pre-training:

```
./tools/contrastive_train.sh
```

Then for fine-tuning:

```
./tools/downstream_train.sh
```

Finally, to compute the IoU metrics use:

```
./tools/eval_train.sh
```

# Citation

If you use this repo, please cite : https://www.mdpi.com/2072-4292/16/21/3984

```
@Article{cloudspam,
AUTHOR = {Mahmoudi Kouhi, Reza and Stocker, Olivier and Giguère, Philippe and Daniel, Sylvie},
TITLE = {CLOUDSPAM: Contrastive Learning On Unlabeled Data for Segmentation and Pre-Training Using Aggregated Point Clouds and MoCo},
JOURNAL = {Remote Sensing},
VOLUME = {16},
YEAR = {2024},
NUMBER = {21},
ARTICLE-NUMBER = {3984},
URL = {https://www.mdpi.com/2072-4292/16/21/3984},
ISSN = {2072-4292},
DOI = {10.3390/rs16213984}
}
```


