import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test
        imgs = [os.path.join(rppt, img) for img in os.listdir(root)]
        
#         test1:data/test1/8973.jpg
#         train:data/train/cat.10004.jpg
        if self.test:
#             如果为测试集
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
#             如果为训练集
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)
        
        if self.test:
            self.imgs = imgs
        elif train:
#             训练集
            self.imgs = imgs[:int(0.7*imgs_num)]
        else:
#             验证集
            self.imgs = imgs[int(0.7*imgs_num):]
        if transforms is None:
            normalize = T.normalize(mean=[0.485, 0.456, 0.406],
                                   std = [0.229, 0.224, 0.225])
            if self.test or not train:
                self.transforms = T.Compose([
                    T.resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
#             训练集
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomReSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
        def __getitem__(self, index):
            img_path = self.imgs[index]
            if self.test:
                label = int(self.imgs[index].split('.')[-2].split('/')[-1])
            else:
                label = 1 if 'dog' in img_path.split('/')[-1] else 0
            data = Image.open(img_path)
            data = self.transforms(data)
            return data, label
        def __len__(self):
            return len(self.imgs)

train_dataset = DogCat(train_data_root, train=True)
trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=num_workers)

