import numpy as np
import sklearn
import torch
from keras import metrics
from scipy.ndimage import sobel  # 用于边缘检测的Sobel滤波器

from medpy import metric
from numpy import mean, std, var
import nibabel as nib
from torchvision.transforms import transforms as T
# import unet
# import Unet3d
from torch import optim
from dataset3dnii_cut_wall import LiverDataset
from torch.utils.data import DataLoader
import myLoss
# import imshow
from PIL import Image
import nrrd
# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import convnext_3d_GCN

x_transform = T.Compose([
    T.ToTensor(),
])

y_transform = T.ToTensor()

# 测试
class LiverDataset_flip(object):
    pass

x'x'x
def test():
    model = convnext_3d_GCN.ConvNeXt(in_chans=1, depths=[1, 1, 3, 1], dims=[16, 32, 64, 128]).to(device)

    # model = Unet3d.UNet(1, 1).to(device)
    # 导入待测试的权重
    model.load_state_dict(torch.load('xxx.pth', map_location='cpu'))
    # 导入测试集数据
    liver_dataset = LiverDataset("xxx", transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(liver_dataset)
    step = 0
    epoch_loss = 0

    dices = []
    jces = []
    hdes = []
    asdes = []
    acces = []
    senes = []
    spes = []
    i = 0
    for x, y in dataloaders:
        inputs = x.to(device)
        labels = y.to(device)
        outputs = model(inputs)
        outputs[outputs < 0.5] = 0
        outputs[outputs >= 0.5] = 1
        i += 1

        dice, jc, hd, asd, acc, sen, spe = calculate_metric_percase(outputs, labels)

        dices.append(dice)
        jces.append(jc)
        hdes.append(hd)
        asdes.append(asd)
        acces.append(acc)
        senes.append(sen)
        spes.append(spe)


        # 计算损失函数
        # loss = torch.nn.BCELoss()(outputs, labels)
        loss = myLoss.BinaryDiceLoss()(outputs, labels)
        step += 1
        str = "%d,test_loss:%0.3f" % (step, loss.item())
        epoch_loss += loss.item()
        print(str + '\n')
    print("epoch_loss:%0.3f" % epoch_loss + '\n' + "mean_loss:%0.3f" % (epoch_loss / step))
    print('dice {}'.format(mean(dices)))
    print('jc {}'.format(mean(jces)))
    print('hd {}'.format(mean(hdes)))
    print('asd {}'.format(mean(asdes)))
    print('acc {}'.format(mean(acces)))
    print('sen {}'.format(mean(senes)))
    print('spe {}'.format(mean(spes)))



def calculate_metric_percase(output, label):
    pred = output.cpu().numpy()
    gt = label.cpu().numpy()
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    acc = (pred == gt).sum()/(pred.shape[1] * pred.shape[2] * pred.shape[3] * pred.shape[4])
    sen = metric.binary.sensitivity(pred, gt)
    spe = metric.binary.specificity(pred, gt)
    return dice, jc, hd, asd, acc, sen, spe

if __name__ == '__main__':
    with torch.no_grad():
        test()
