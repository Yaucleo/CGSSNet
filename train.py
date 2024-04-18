import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torchvision.transforms import transforms as T
import argparse  # argparse模块的作用是用于解析命令行参数，例如python parseTest.py input.txt --port=8080
# import Unet3d_multi_depth
from torch import optim
from dataset3dnii import LiverDataset
from torch.utils.data import DataLoader
import myLoss
import convnext_3d_GCN
# import vnet
CUDA_LAUNCH_BLOCKING = 1

device = torch.device("cuda")
# 是否使用gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_transform = T.Compose([
    T.ToTensor()
])
# mask只需要转换为tensor
y_transform = T.ToTensor()

def train_model(model, criterion1, criterion2, optimizer, dataload, num_epochs=200):
    # 将历史最小的loss（取值范围是[0,1]）初始化为最大值1
    min_testloiss = 1
    for epoch in range(num_epochs):
        # 5个epoch不优化则降低学习率
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
                                                   threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                   eps=1e-06)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # 训练集数据个数
        dataset_size = len(dataload.dataset)
        # 每个epoch的loss
        epoch_loss = 0
        # 当前epoch的当前计算数据序号
        step = 0
        i = 0
        # 遍历数据集，batch_size=1， 共进行num_epochs次
        for x, y in dataload:

            optimizer.zero_grad()

            inputs = x.to(device)
            labels = y.to(device)
            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion1(outputs, labels)
            # loss.requires_grad_(True)
            # 梯度下降,计算出梯度
            loss.backward()
            # 对所有的参数进行一次更新
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
        str = "epoch %d loss:%0.3f" % (epoch, epoch_loss)
        print(str)
        # 记录训练结果
        f = open('training_log_single.txt', 'a')
        f.write(str +'\n')
        f.close()
        # 更新保存的当前权重，用于在验证集测试
        torch.save(model.state_dict(), 'weights_single.pth')
        # if (epoch % 3 == 0):
            # torch.save(model.state_dict(), 'weights_%d.pth' % epoch)  # 返回模型的所有内容
        with torch.no_grad():
            # 使用保存的当前权重计算验证集上的损失
            testloss = test(epoch)
        # loss比历史最小loss小时独立保存
        if testloss < min_testloiss:
            torch.save(model.state_dict(), 'test_weights%0.3f.pth' % testloss)
            min_testloiss = testloss
    return model


# 训练模型
def train():
    model = convnext_3d_GCN.ConvNeXt(in_chans=1, depths=[1, 1, 3, 1], dims=[16, 32, 64, 128]).to(device)
    batch_size = 1
    # 两种损失函数
    criterion2 = torch.nn.BCELoss()
    criterion1 = myLoss.BinaryDiceLoss()  # 指定损失函数为自定义
    # 梯度下降的优化器，使用默认学习率
    optimizer = optim.Adam(model.parameters())  # model.parameters():Returns an iterator over module parameters
    # 加载数据集
    liver_dataset = LiverDataset("2022_train", transform=x_transform, target_transform=y_transform)
    dataloader = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True)
    # 开始训练
    train_model(model, criterion1, criterion2, optimizer, dataloader)
# 测试
def test(e):
    model = convnext_3d_GCN.ConvNeXt(in_chans=1, depths=[1, 1, 3, 1], dims=[16, 32, 64, 128]).to(device)

    model.load_state_dict(torch.load('weights_single.pth', map_location='cpu'))
    # 使用测试集数据进行测试
    liver_dataset = LiverDataset("xxx", transform=x_transform, target_transform=y_transform)

    dataloaders = DataLoader(liver_dataset)
    step = 0
    sumloss = 0
    for x, y in dataloaders:
        inputs = x.to(device)
        labels = y.to(device)
        outputs = model(inputs)  # 前向传播
        loss = myLoss.BinaryDiceLoss()(outputs, labels)
        sumloss += loss
        step += 1
        str = "%d,test_loss:%0.3f" % (step, loss.item())
        print(str+'\n')
    print("meanloss:%0.3f" % (sumloss/step))
    log("meanloss:%0.3f" % (sumloss/step))
    return sumloss/step

# 保存信息到日志中
def log(str):
    f = open('training_log_single.txt', 'a')
    f.write(str + '\n')
    f.close()

if __name__ == '__main__':
    with torch.cuda.device(0):
        train()

