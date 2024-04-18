
import torch
import torch.utils.data as data
import os
import nrrd
import nibabel as nib
import numpy as np
class LiverDataset(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None):  # root表示图片路径
        n = len(os.listdir(root)) // 3
        imgs = []
        for i in range(1, n + 1):

            img = os.path.join(root, "enhanced_" + str(i)+'.nii.gz')
            mask = os.path.join(root, "atriumSegImgMO_" + str(i)+'.nii.gz')
            imgs.append([img, mask])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = nib.load(x_path).get_fdata()
        img_y = nib.load(y_path).get_fdata()/420

        img_x = img_x.unsqueeze(0)
        img_y = img_y.unsqueeze(0)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
