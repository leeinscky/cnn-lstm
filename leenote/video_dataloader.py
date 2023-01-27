# How upload sequence of image on video-classification? 代码来源：https://discuss.pytorch.org/t/how-upload-sequence-of-image-on-video-classification/24865
# 定义dataloader实现每次读取一段连续的图片序列，而不是一张图片，这样可以用于视频分类。
# conda activate cnn-lstm && cd /home/zl525/code/cnn-lstm/leenote && python video_dataloader.py

import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image


class MySampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):        
        indices = []
        print(f'[MySampler] end_idx={end_idx}, seq_length={seq_length}, len(end_idx)={len(end_idx)}')
        for i in range(len(end_idx)-1):
            start = end_idx[i]
            end = end_idx[i+1] - seq_length
            print(f'[MySampler] i={i}, start={start}, end={end}')
            # indices.append(torch.arange(start, end)) # 原始逻辑
            indices.append(torch.arange(start, end, step=seq_length)) # 修改后逻辑
            print(f'[MySampler] 当前循环-indices={indices}')
        indices = torch.cat(indices)
        print(f'[MySampler] self.indices = indices={indices}')
        self.indices = indices
        
    def __iter__(self):
        
        ######## 原始逻辑：打乱self.indices的顺序 ########
        # random_index = torch.randperm(len(self.indices))
        # indices = self.indices[random_index] # randperm 生成随机排列，返回一个新的tensor. 整行代码指的是：随机打乱self.indices的顺序
        # print(f'[MySampler] __iter__-indices={indices}, random_index = {random_index}')
        # print(f'[MySampler] iter(indices.tolist())={iter(indices.tolist())}')
        
        ######## 修改后逻辑：不打乱self.indices的顺序 ########
        indices = self.indices
        print(f'[MySampler] __iter__-indices={indices}')
        print(f'[MySampler] iter(indices.tolist())={iter(indices.tolist())}')
        return iter(indices.tolist()) # iter() 函数用来生成迭代器, 返回一个迭代器对象
    
    def __len__(self):
        return len(self.indices)


class MyDataset(Dataset):
    def __init__(self, image_paths, seq_length, transform, length):
        self.image_paths = image_paths
        self.seq_length = seq_length
        self.transform = transform
        self.length = length
        
        # image_paths = class_image_paths=[
            # ('./lstm_dataset_test/bowling/person1/image_00001.jpg', 0), 
            # ('./lstm_dataset_test/bowling/person1/image_00002.jpg', 0), 
            # ('./lstm_dataset_test/bowling/person1/image_00003.jpg', 0), 
            # ('./lstm_dataset_test/bowling/person1/image_00004.jpg', 0), 
            # ('./lstm_dataset_test/bowling/person1/image_00005.jpg', 0), 
            # ('./lstm_dataset_test/bowling/person1/image_00006.jpg', 0), 
            # ('./lstm_dataset_test/bowling/person1/image_00007.jpg', 0), 
            # ('./lstm_dataset_test/bowling/person0/image_00001.jpg', 0), 
            # ('./lstm_dataset_test/bowling/person0/image_00002.jpg', 0), 
            # ('./lstm_dataset_test/bowling/person0/image_00003.jpg', 0), 
            # ('./lstm_dataset_test/bowling/person0/image_00004.jpg', 0), 
            # ('./lstm_dataset_test/bowling/person0/image_00005.jpg', 0), 
            # ('./lstm_dataset_test/bowling/person0/image_00006.jpg', 0), 
            # ('./lstm_dataset_test/ApplyEyeMakeup/person1/image_00001.jpg', 1), 
            # ('./lstm_dataset_test/ApplyEyeMakeup/person1/image_00002.jpg', 1), 
            # ('./lstm_dataset_test/ApplyEyeMakeup/person1/image_00003.jpg', 1), 
            # ('./lstm_dataset_test/ApplyEyeMakeup/person1/image_00004.jpg', 1), 
            # ('./lstm_dataset_test/ApplyEyeMakeup/person1/image_00005.jpg', 1), 
            # ('./lstm_dataset_test/ApplyEyeMakeup/person1/image_00006.jpg', 1), 
            # ('./lstm_dataset_test/ApplyEyeMakeup/person1/image_00007.jpg', 1), 
            # ('./lstm_dataset_test/ApplyEyeMakeup/person1/image_00008.jpg', 1), 
            # ('./lstm_dataset_test/ApplyEyeMakeup/person0/image_00001.jpg', 1), 
            # ('./lstm_dataset_test/ApplyEyeMakeup/person0/image_00002.jpg', 1), 
            # ('./lstm_dataset_test/ApplyEyeMakeup/person0/image_00003.jpg', 1), 
            # ('./lstm_dataset_test/ApplyEyeMakeup/person0/image_00004.jpg', 1)]
        
    def __getitem__(self, index):
        start = index
        end = index + self.seq_length
        print('[MyDataset] index={}, Getting images from {} to {}'.format(index, start, end))
        indices = list(range(start, end))
        print(f'indices={indices}')
        images = []
        for i in indices:
            image_path = self.image_paths[i][0]
            print(f'[MyDataset] 遍历indices，当前循环 i={i}，image_path={image_path}')
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image) # 将seq_length个图片组成一个list
        x = torch.stack(images) # x是seq_length个图片的tensor
        print(f'[MyDataset] x.shape={x.shape}')
        y = torch.tensor([self.image_paths[start][1]], dtype=torch.long)
        print(f'[MyDataset] y={y}, y.shape={y.shape}')
        
        return x, y
    
    def __len__(self):
        return self.length


if __name__ == '__main__':
    root_dir = './lstm_dataset_test'
    class_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]
    print(f'class_paths={class_paths}')

    class_image_paths = []
    end_idx = []
    for c, class_path in enumerate(class_paths):
        print(f'c={c}, class_path={class_path}')
        for d in os.scandir(class_path):
            print(f'd.path={d.path}')
            if d.is_dir:
                paths = sorted(glob.glob(os.path.join(d.path, '*.jpg'))) # 注意根据实际情况改为jpg or png
                # print(f'paths-1={paths}')
                # Add class idx to paths
                paths = [(p, c) for p in paths]
                # print(f'paths-2={paths}')
                class_image_paths.extend(paths)
                # print(f'class_image_paths={class_image_paths}')
                end_idx.extend([len(paths)])
                print(f'end_idx={end_idx}')

    end_idx = [0, *end_idx]
    print(f'2-end_idx={end_idx}')
    end_idx = torch.cumsum(torch.tensor(end_idx), 0) # 将end_idx转换为累加的形式，cumsum是累加函数，0表示按列累加，1表示按行累加，这里是按列累加，即每一列的元素相加，最后得到的是一个一维的向量。
    print(f'3-end_idx={end_idx}, end_idx.shape={end_idx.shape}')
    seq_length = 2

    sampler = MySampler(end_idx, seq_length)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    dataset = MyDataset(
        image_paths=class_image_paths,
        seq_length=seq_length,
        transform=transform,
        length=len(sampler))

    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler
    )

    for data, target in loader:
        print(data.shape)