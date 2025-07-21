## Environment Setup
### step 1. Install environment or Docker Pull
Install environment
```
conda create --name ProtoOcc python=3.7.11
conda activate ProtoOcc
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.3
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0

sudo apt-get install python3-dev 
sudo apt-get install libevent-dev
sudo apt-get groupinstall 'development tools'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda
pip install pycuda

pip install lyft_dataset_sdk
pip install networkx==2.2
pip install numba==0.53.0
pip install numpy==1.23.5
pip install nuscenes-devkit
pip install plyfile
pip install scikit-image
pip install tensorboard
pip install trimesh==2.35.39
pip install setuptools==59.5.0
pip install yapf==0.40.1
```
Docker pull command
```
docker pull junghokim1/protoocc:python3.7-torch-1.10.0-cu111
sudo docker run -it -e DISPLAY=unix$DISPLAY --gpus all --ipc=host -v /{src}:/{tar} -e XAUTHORITY=/tmp/.docker.xauth --name ProtoOcc junghokim1/protoocc:python3.7-torch-1.10.0-cu111 /bin/bash
```
#### After setting up the environment or pulling the Docker image, run `git clone` and `pip install`.
```
git clone https://github.com/SPA-junghokim/ProtoOcc
cd ProtoOcc

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc4
pip install -v -e .

cd ../projects
pip install -v -e .
cd ..
```

### Step 2. Download and unzip the [nuScenes dataset](https://www.nuscenes.org/download) (including panoptic files), and get the gts folder from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction).
For auxiliary task (perspective semantic segmentation), replace `v1.0-trainval/category.json` within the received folder with the `category.json` in `./data/nuscenes/v1.0-trainval`, and move `v1.0-trainval/panoptic.json` from the received folder to `./data/nuscenes/v1.0-trainval`.

### step 3. Prepare nuScenes dataset as below:
```shell script
└── ProtoOcc/
    └── data
        └── nuscenes
            ├── v1.0-trainval
                ├── panoptic.json
                ├── category.json # (from nuscenes panoptic)
                ├── ...
            ├── sweeps 
            ├── samples
            ├── panoptic
            └── gts 
```


### step 4. Preprocess for training

Create the pkl or download [Here](https://drive.google.com/drive/folders/1aiG4wmsj4Q7cJBQ4-H1lrG8wiFIhoa-W?usp=drive_link):
```shell script
python tools/create_data_bevdet.py
```

Run code below for `pc_panoptic`
```shell script
python tools/data_converter/prepare_panoptic.py
```


### step 5. Download [ckpts](https://drive.google.com/drive/folders/1e459AGnjwtatnakv2kyOR2beweJTk03e?usp=sharing) to `ProtoOcc/ckpts/`:

### The final directory should be organized as follows 
```shell script
└── ProtoOcc/
    ├── data
        └── nuscenes
            ├── v1.0-trainval 
            ├── sweeps  
            ├── samples
            ├── panoptic
            ├── pc_panoptic
            ├── gts 
            ├── bevdetv2-nuscenes_infos_train.pkl 
            └── bevdetv2-nuscenes_infos_val.pkl
    ├── ckpts
        ├── bevdet-r50-4d-depth-cbgs_depthnet_modify.pth # (Renamed 'depth_net' in 'state_dict' for pretrained weights)
        ├── bevdet-r50-4dlongterm-stereo-cbgs.pth
    ├── doc
    ├── mmdetection3d 
    ├── projects
    ├── requirements
    ├── tools
    ├── plot
    └── README.md
```

---
## SemanticKITTI

To prepare for SemanticKITTI dataset, please download the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (including color, velodyne laser data, and calibration files) and the annotations for Semantic Scene Completion from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download). Put all `.zip` files under `OccFormer/data/SemanticKITTI` and unzip these files. Then you should get the following dataset structure:
```shell script
└── ProtoOcc
    ├── data/
        └── SemanticKITTI/
            └── dataset/
                ├── sequences
                    ├── 00
                        ├── calib.txt
                        ├── poses.txt
                        ├── calib.txt
                        ├── labels/
                        ├── image_2/
                        ├── image_3/
                        ├── velodyne/
                        └── voxels/
                    ├── 01
                    ├── 02
                    ├── ...
                    └── 10
                └── labels
    └── ckpts
        ├── efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth
        └── occformer_kitti.pth # (this file is from "https://github.com/zhangyp15/OccFormer")
```

Preprocess the annotations for semantic scene completion:
```bash
python projects/mmdet3d_plugin/tools/kitti_process/semantic_kitti_preprocess.py --kitti_root data/SemanticKITTI --kitti_preprocess_root data/SemanticKITTI --data_info_path projects/mmdet3d_plugin/tools/kitti_process/semantic-kitti.yaml
```
