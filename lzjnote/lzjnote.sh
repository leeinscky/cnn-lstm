# ucf101 数据集下载部分视频：
    https://www.kaggle.com/datasets/pevogam/ucf101?resource=download

# 测试命令 
cd code/cnn-lstm/
python main.py --batch_size 8 --n_epochs 100 --num_workers 0  --annotation_path ./data/annotation/ucf101_01.json --video_path ./data/image_data/  --dataset ucf101 --sample_size 150 --lr_rate 1e-4 --n_classes 2