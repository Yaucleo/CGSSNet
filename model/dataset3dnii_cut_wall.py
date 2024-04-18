
import torch
import torch.utils.data as data
import os
import nrrd
import nibabel as nib
import numpy as np
import cv2
from scipy.ndimage import sobel  # 用于边缘检测的Sobel滤波器
from scipy.ndimage import morphology
import convnext_3d_GCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 所有子类应该override __len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)
class LiverDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, root, transform=None, target_transform=None):  # root表示图片路径
        # os.listdir(path)返回指定路径下的文件和文件夹列表。除以2指路径中一份原图和一份标签当做一组数据
        n = len(os.listdir(root)) // 3
        # 列表，放入每一组数据中原图和标签的路径
        imgs = []

        # 模型载入
        model = convnext_3d_GCN.ConvNeXt(in_chans=1, depths=[1, 1, 3, 1], dims=[16, 32, 64, 128]).to(device)
        # model = Unet3d.UNet(1, 1).to(device)
        # 导入待测试的权重
        model.load_state_dict(torch.load('11.29_lasnet/test_weights0.027.pth', map_location='cpu'))

        #
        for i in range(1, n + 1):
            # os.path.join(path1[,path2[,......]]):将多个路径组合后返回
            # 原图的路径
            img = os.path.join(root, "enhanced_"+str(i)+'.nii.gz')
            # img = os.path.join(root, "%03d_lgemri.nrrd" % i)
            # 标签的路径
            # mask = os.path.join(root, "atriumSegImgMO_"+ str(i)+'.nii.gz')

            mask = os.path.join(root, "scarSegImgM_"+ str(i)+'.nii.gz')
            # mask = os.path.join(root, "atriumSegImgMO_"+ str(i)+'.nii.gz')


            # mask = os.path.join(root, "%03d_lawall.nrrd" % i)
            # 原图和标签的路径放到数组中作为一组数据，存入imgs列表
            imgs.append([img, mask])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.model = model

    # 重写getitem，根据索引index对应的一组数据，返回原图和标签的矩阵
    def __getitem__(self, index):
        # 取出原图和标签的路径img, mask
        x_path, y_path = self.imgs[index]
        img_x = nib.load(x_path).get_fdata()
        img_y = nib.load(y_path).get_fdata()


        img_x = img_x.unsqueeze(0)
        inputs = img_x.unsqueeze(0)

        img_y = img_y.unsqueeze(0)


        # 图像处理
        # 模型对原图进行腔体分割
        inputs = inputs.to(device)
        outputs = self.model(inputs)

        outputs[outputs < 0.5] = 0
        outputs[outputs >= 0.5] = 1

        outputs = outputs.cpu().detach().numpy()
        outputs = outputs.squeeze(0)
        outputs = outputs.squeeze(0)

        # 轮廓提取
        # 使用Sobel滤波器进行边缘检测
        edges_x = sobel(outputs, axis=0)
        edges_y = sobel(outputs, axis=1)
        edges_z = sobel(outputs, axis=2)

        # 计算总边缘强度
        total_edges = np.sqrt(edges_x ** 2 + edges_y ** 2 + edges_z ** 2)
        total_edges[total_edges != 0] = 1



        # 定义一个3x3x3的核
        structure_3 = np.ones((3, 3, 3), dtype=bool)
        structure_5 = np.ones((5, 5, 5), dtype=bool)


        # 有问题
        # 使用膨胀操作 5
        total_edges_dila = morphology.binary_dilation(total_edges, structure=structure_5)

        # 腐蚀 3
        total_edges_ero = morphology.binary_erosion(total_edges, structure=structure_3)

        result_edges = total_edges_dila + total_edges_ero

        # 需要再次验证是否真的是一个环形区域
        result_edges[result_edges != 1] = 0



        # total_edges[total_edges != 0] = 1


        # numpy_data = numpy_data.squeeze(0)




        # 区域
        total_edges = torch.Tensor(result_edges).unsqueeze(0)



        # 限制区域
        img_x = img_x * total_edges



        # img = torch.squeeze(img_x, 0)
        # img = nib.Nifti1Image(img.cpu().numpy(), affine=np.eye(4))  # affine参数通常是单位矩阵
        #
        # # 保存为.nii文件
        # nib.save(img, 'aaa.nii')







        total = np.sum(total_edges.numpy())
        # # 强度归一化
        #
        img_x = img_x.numpy()
        a = np.sum(img_x)
        I = img_x / (total/a)
        # Z-score
        mean_I = np.sum(I) / total
        st = np.std(I[I != 0])
        I_z = (I - mean_I) / st
        img_x = torch.tensor(I_z)
        #
        # diff1 = np.sum(normalized_image.numpy() - img_x)


        # img = torch.squeeze(normalized_image, 0)
        #
        #
        # img = nib.Nifti1Image(img.cpu().numpy(), affine=np.eye(4))  # affine参数通常是单位矩阵
        #
        # nib.save(img, 'bbb.nii')

        # 另一种强度比归一化方法
        # 计算图像的最大和最小强度值
        # imgx = img_x.numpy()
        # max_value = np.max(imgx[imgx != 0])
        # min_value = np.min(imgx[imgx != 0])
        # imgx = imgx - min_value
        # imgx[imgx < 0] = 0
        # normalized_image = ( imgx/ (max_value - min_value)) * (255 - 0) + 0
        # img_x = torch.tensor(normalized_image)
        # diff2 = np.sum(imgx - normalized_image)
        # print(diff2)

        #
        # img = torch.squeeze(torch.tensor(normalized_image), 0)
        # # #
        # # #
        # img = nib.Nifti1Image(img.cpu().numpy(), affine=np.eye(4))  # affine参数通常是单位矩阵
        #
        # #
        # # # 保存为.nii文件
        # nib.save(img, 'bbb.nii')

        # img_y = img_y * total_edges
        # elif
        # if self.transform is not None:
        #     img_x = torch.Tensor(img_x[:, :, 0:64])
        #     # img_x = self.transform(img_x[:, :, 3:83])
        #     # img_x = self.transform(img_x[0:576:t, 0:576:t, 4:84])
        #     # 增加图片的通道数的维度，符合三维卷积运算的输入标准
        #     img_x = img_x.unsqueeze(0)
        # if self.target_transform is not None:
        #     img_y = torch.Tensor(img_y)
        #     img_y = img_y[:,:,0:64]
        #
        #     # img_y = np.array(img_y)
        #     # img_y = self.transform(img_y[:, :, 3:83])
        #
        #     # img_y = self.target_transform(img_y[32:544:t, 32:544:t, 4:84])
        #
        #     # img_y = self.target_transform(img_y[0:576:t, 0:576:t, 4:84])
        #     img_y = img_y.unsqueeze(0)
        return img_x, img_y

    # 重写len，返回数据集的数据数量
    def __len__(self):
        return len(self.imgs)
