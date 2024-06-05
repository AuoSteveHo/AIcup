# AIcup - 跨相機多目標車輛追蹤競賽-參數組-程式碼上傳
## Team 4997

##安裝

** 開發環境 python 3.8
** 依循官方教學

**1.** 
```shell
conda create -n botsort python=3.8
conda activate botsort
```

**2.** 
```shell
pip install numpy
```

**3.** 
```shell
pip install -r requirements.txt
```

**4.** 
```shell
pip install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

**5.** 
```shell
# Cython-bbox
pip install cython_bbox

# faiss cpu / gpu
pip install faiss-cpu
pip install faiss-gpu
```

##資料準備
```shell
python fast_reid/datasets/generate_AICUP_patches.py --data_path 32_33_train_v2/train

python yolov7/tools/AICUP_to_YOLOv7.py --AICUP_dir 32_33_train_v2/train --YOLOv7_dir datasets/AI_CUP_MCMOT_dataset/yolo
```
##模型訓練
```shell
python fast_reid/tools/train_net.py --config-file fast_reid/configs/AICUP/bagtricks_R50-ibn.yml MODEL.DEVICE "cuda:0"

python yolov7/train.py --device 0 --batch-size 8 --epochs 50 --workers 2 --data yolov7/data/AICUP.yaml --img 640 640 --cfg yolov7/cfg/training/yolov7-AICUP.yaml --weights 'pretrained/yolov7-e6e.pt' --name yolov7-AICUP --hyp data/hyp.scratch.custom.yaml --save_period 1

python yolov7/train_aux.py --device 0 --batch-size 8 --epochs 50 --workers 2 --data yolov7/data/AICUP.yaml --img 640 640 --cfg yolov7/cfg/training/yolov7-w6-AICUP.yaml --weights 'pretrained/yolov7-e6e.pt' --name yolov7-w6-AICUP --hyp data/hyp.scratch.custom.yaml --save_period 1
```
##模型測試
```shell
python tools/mc_demo_yolov7.py --weights runs/train/yolov7-AICUP/weights/best.pt --source 32_33_train_v2/train/images/0924_150000_151900 --device 1 --name 0924_150000_151900 --fuse-score --agnostic-nms --with-reid --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth

track_all_timestamps.sh --weights runs/train/yolov7-AICUP/weights/best.pt --source-dir 32_33_train_v2/train/images --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth

track_all_timestamps.sh --weights runs/train/yolov7-w6-AICUP/weights/best.pt --source-dir 32_33_train_v2/train/images --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth
```
##模型評估
```shell
python tools/datasets/AICUP_to_MOT15.py --AICUP_dir 32_33_train_v2/ --MOT15_dir  MOT15_dir/

python tools/evaluate.py --gt_dir MOT15_dir/ --ts_dir runs/detect/yolov7-AICUP/detect_result/

python tools/evaluate.py --gt_dir MOT15_dir/ --ts_dir runs/detect/yolov7-w6-AICUP/detect_result/

python tools/evaluate.py --gt_dir 1/ --ts_dir 2/
```
##影像增量
```shell
python augmentation.py

python fast_reid/datasets/generate_AICUP_patches.py --data_path 32_33_train_v2/augmentation

python yolov7/tools/AICUP_to_YOLOv7.py --AICUP_dir 32_33_train_v2/augmentation --YOLOv7_dir datasets/AI_CUP_MCMOT_dataset/yolo_augmentation

python fast_reid/tools/train_net.py --config-file fast_reid/configs/AICUP/bagtricks_R50-ibn.yml MODEL.DEVICE "cuda:0"

python yolov7/train.py --device 0 --batch-size 8 --epochs 50 --workers 2 --data yolov7/data/AICUP_augmentation.yaml --img 640 640 --cfg yolov7/cfg/training/yolov7-AICUP.yaml --weights 'pretrained/yolov7-e6e.pt' --name yolov7-AICUP-augmentation --hyp yolov7/data/hyp.scratch.p5.yaml --save_period 1

track_all_timestamps.sh --weights runs/train/yolov7-AICUP-augmentation/weights/best.pt --source-dir 32_33_train_v2/train/images --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth

python tools/evaluate.py --gt_dir MOT15_dir/ --ts_dir runs/detect/yolov7-AICUP-augmentation/detect_result/

python tools/evaluate.py --gt_dir MOT15_dir/ --ts_dir runs/detect/yolov7-AICUP-augmentation3/
```
##最終測試
```shell
track_all_timestamps.sh --weights runs/train/yolov7-AICUP-augmentation/weights/best.pt --source-dir 32_33_AI_CUP_testdataset/AI_CUP_testdata/images --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth

track_all_timestamps.sh --weights runs/train/yolov7-AICUP-augmentation/weights/last.pt --source-dir 32_33_AI_CUP_testdataset/AI_CUP_testdata/images --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth
```
##參數調整
```shell
track_all_timestamps.sh --weights runs/train/yolov7-AICUP-augmentation/weights/best.pt --source-dir test --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth
track_all_timestamps.sh --weights runs/train/yolov7-AICUP-augmentation/weights/last.pt --source-dir test --device 1 --fast-reid-config fast_reid/configs/AICUP/bagtricks_R50-ibn.yml --fast-reid-weights logs/AICUP_115/bagtricks_R50-ibn/model_0058.pth
python tools/evaluate.py --gt_dir test2/ --ts_dir runs/detect/0903_150000_151900/
```
