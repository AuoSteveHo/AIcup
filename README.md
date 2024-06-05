# AIcup 跨相機多目標車輛追蹤競賽-參數組 - Team 4997 程式碼上傳
**開發環境為python 3.8 其他依循官方教學**

google drive連結： https://drive.google.com/drive/folders/1S0d1Ha01reJCOBupABaYDo3hjgMQRqg5?usp=drive_link
## 模型
[model_058.pth](https://github.com/AuoSteveHo/AIcup/releases/download/untagged-d6594f5d82b2a28cdb30/model_0058.pth)
[last.pt](https://github.com/AuoSteveHo/AIcup/releases/download/untagged-66decab18a1dfdedb6ab/last.pt)

## 安裝
```shell
conda create -n AIcup python=3.8
conda activate AIcup
```

```shell
pip install numpy
```

```shell
pip install -r requirements.txt
```

```shell
pip install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

```shell
# Cython-bbox
pip install cython_bbox

# faiss cpu / gpu
pip install faiss-cpu
pip install faiss-gpu
```
## 套件
- absl-py==2.0.0
- appdirs==1.4.4
- beautifulsoup4==4.12.3
- cachetools==5.3.3
- certifi==2020.12.5
- charset-normalizer==3.3.2
- click==7.1.2
- colorama==0.3.5
- cycler==0.12.1
- Cython==3.0.9
- cython-bbox==0.1.5
- distro==1.9.0
- docker-pycreds==0.4.0
- dominate==2.5.1
- easydict==1.13
- faiss-cpu==1.8.0
- filelock==3.9.1
- filterpy==1.1.0
- flatbuffers==24.3.7
- fonttools==4.51.0
- gdown==5.1.0
- gitdb==4.0.9
- GitPython==3.1.33
- google-auth==2.9.0
- google-auth-oauthlib==0.4.6
- grpcio==1.59.3
- h5py==3.9.0
- idna==3.6
- imageio==2.27.0
- importlib-metadata==7.1.0
- joblib==1.3.2
- jsonpatch==1.33
- jsonpointer==2.4
- kiwisolver==1.4.5
- lap==0.4.0
- lazy-loader==0.4
- loguru==0.6.0
- Markdown==3.6
- MarkupSafe==2.1.1
- matplotlib==3.3.2
- motmetrics==1.4.0
- networkx==2.8.8
- ninja==1.9.0.post1
- numpy==1.24.3
- oauthlib==3.2.2
- onnx==1.8.1
- onnx-simplifier==0.3.5
- onnxoptimizer==0.3.9
- onnxruntime==1.8.0
- opencv-python==4.8.0.74
- packaging==23.1
- pandas==2.0.3
- pathtools==0.1.2
- Pillow==9.5.0
- prettytable==3.9.0
- protobuf==3.20.0
- psutil==5.9.3
- py-cpuinfo==9.0.0
- pyasn1==0.5.1
- pyasn1-modules==0.3.0
- pycocotools==2.0
- pyparsing==3.1.2
- PySocks==1.7.1
- python-dateutil==2.9.0
- pytz==2020.1
- PyWavelets==1.4.1
- PyYAML==6.0
- pyzmq==24.0.0
- requests==2.29.0
- requests-oauthlib==2.0.0
- rsa==4.9
- scikit-build==0.17.6
- scikit-image==0.21.0rc1
- scikit-learn==1.3.2
- scipy==1.10.0
- seaborn==0.13.0
- sentry-sdk==1.10.0
- setproctitle==1.2.2
- six==1.9.0
- smmap==5.0.1
- soupsieve==2.5
- tabulate==0.9.0
- tensorboard==2.9.0
- tensorboard-data-server==0.6.1
- tensorboard-plugin-wit==1.8.1
- termcolor==2.4.0
- thop==0.1.1.post2209072238
- threadpoolctl==3.4.0
- tifffile==2023.1.23.1
- tomli==2.0.1
- torch==1.8.1+cu101
- torchvision==0.9.1+cu101
- tornado==6.2b1
- tqdm==4.65.0
- typing-extensions==4.9.0
- tzdata==2022.1
- ultralytics==8.1.24
- urllib3==1.26.17
- visdom==0.2.1
- wandb==0.15.10
- wcwidth==0.2.9
- websocket-client==0.50.0
- werkzeug==3.0.0
- win32-setctime==1.1.0
- wincertstore==0.2
- xmltodict==0.13.0
- yacs==0.1.8
- zipp==3.9.1

## 資料準備
```shell
python fast_reid/datasets/generate_AICUP_patches.py --data_path 32_33_train_v2/train

python yolov7/tools/AICUP_to_YOLOv7.py --AICUP_dir 32_33_train_v2/train --YOLOv7_dir datasets/AI_CUP_MCMOT_dataset/yolo
```
## 模型訓練
```shell
python fast_reid/tools/train_net.py --config-file fast_reid/configs/AICUP/bagtricks_R50-ibn.yml MODEL.DEVICE "cuda:0"

python yolov7/train.py --device 0 --batch-size 8 --epochs 50 --workers 2 --data yolov7/data/AICUP.yaml --img 640 640 --cfg yolov7/cfg/training/yolov7-AICUP.yaml --weights 'pretrained/yolov7-e6e.pt' --name yolov7-AICUP --hyp data/hyp.scratch.custom.yaml --save_period 1

python yolov7/train_aux.py --device 0 --batch-size 8 --epochs 50 --workers 2 --data yolov7/data/AICUP.yaml --img 640 640 --cfg yolov7/cfg/training/yolov7-w6-AICUP.yaml --weights 'pretrained/yolov7-e6e.pt' --name yolov7-w6-AICUP --hyp data/hyp.scratch.custom.yaml --save_period 1
```
## 模型測試
```shell
python tools/mc_demo_yolov7.py --weights runs/train/yolov7-AICUP/weights/best.pt --source 32_33_train_v2/train/images/0924_150000_151900 --device 1 --name 0924_150000_151900 --fuse-score --agnostic-nms --with-reid --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth

track_all_timestamps.sh --weights runs/train/yolov7-AICUP/weights/best.pt --source-dir 32_33_train_v2/train/images --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth

track_all_timestamps.sh --weights runs/train/yolov7-w6-AICUP/weights/best.pt --source-dir 32_33_train_v2/train/images --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth
```
## 模型評估
```shell
python tools/datasets/AICUP_to_MOT15.py --AICUP_dir 32_33_train_v2/ --MOT15_dir  MOT15_dir/

python tools/evaluate.py --gt_dir MOT15_dir/ --ts_dir runs/detect/yolov7-AICUP/detect_result/

python tools/evaluate.py --gt_dir MOT15_dir/ --ts_dir runs/detect/yolov7-w6-AICUP/detect_result/
```
## 影像增量
```shell
python augmentation.py

python fast_reid/datasets/generate_AICUP_patches.py --data_path 32_33_train_v2/augmentation

python yolov7/tools/AICUP_to_YOLOv7.py --AICUP_dir 32_33_train_v2/augmentation --YOLOv7_dir datasets/AI_CUP_MCMOT_dataset/yolo_augmentation

python fast_reid/tools/train_net.py --config-file fast_reid/configs/AICUP/bagtricks_R50-ibn.yml MODEL.DEVICE "cuda:0"

python yolov7/train.py --device 0 --batch-size 8 --epochs 50 --workers 2 --data yolov7/data/AICUP_augmentation.yaml --img 640 640 --cfg yolov7/cfg/training/yolov7-AICUP.yaml --weights 'pretrained/yolov7-e6e.pt' --name yolov7-AICUP-augmentation --hyp yolov7/data/hyp.scratch.p5.yaml --save_period 1

track_all_timestamps.sh --weights runs/train/yolov7-AICUP-augmentation/weights/best.pt --source-dir 32_33_train_v2/train/images --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth

python tools/evaluate.py --gt_dir MOT15_dir/ --ts_dir runs/detect/yolov7-AICUP-augmentation/detect_result/
```
## 最終測試
```shell
track_all_timestamps.sh --weights runs/train/yolov7-AICUP-augmentation/weights/best.pt --source-dir 32_33_AI_CUP_testdataset/AI_CUP_testdata/images --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth

track_all_timestamps.sh --weights runs/train/yolov7-AICUP-augmentation/weights/last.pt --source-dir 32_33_AI_CUP_testdataset/AI_CUP_testdata/images --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth
```
## 參數調整
```shell
track_all_timestamps.sh --weights runs/train/yolov7-AICUP-augmentation/weights/best.pt --source-dir test --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth
track_all_timestamps.sh --weights runs/train/yolov7-AICUP-augmentation/weights/last.pt --source-dir test --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth
python tools/evaluate.py --gt_dir test2/ --ts_dir runs/detect/0903_150000_151900/
```
## 結果上傳
1. 32_33_submission_sample.zip / TEST $\Longrightarrow$ Public Score: 0.466868
2. 20240528_team4997_1.zip / best.pt $\Longrightarrow$ Public Score: 0.789098
3. 20240528_team4997_2.zip / best.pt, conf-thres 0.05, appearance_thresh 0.28 $\Longrightarrow$ Public Score: 0.810234
4. 20240528_team4997_3.zip / last.pt $\Longrightarrow$ Public Score: 0.834429
5. 20240528_team4997_4.zip / last.pt, conf-thres 0.05, appearance_thresh 0.26 $\Longrightarrow$ Public Score: 0.818805
