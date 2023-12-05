# 训练+测试


import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import cv2

# import os

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数
the_epochs = 3  # 训练整批数据的次数
batch_size_train = 50  # 训练集批处理大小为50
batch_size_test = 1000  # 测试集批处理大小为1000
learn_rate = 0.01  # 学习率为0.01
# momentum = 0.5 待用
DOWNLOAD_MNIST = True  # 表示还没有下载数据集，如果数据集下载好了就写False

# 下载mnist手写数据集
train_data = torchvision.datasets.MNIST(
    root='./data/',  # 保存或提取的位置  会放在当前文件夹中
    train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
        torchvision.transforms.Normalize((0.1301,), 0.3088),  # 设置 minist 数据集全局平均值和标准偏差。
    ]),
    download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
)

test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False,  # 表明是测试集
    transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
            torchvision.transforms.Normalize((0.1301,), 0.3088),  # 设置 minist 数据集全局平均值和标准偏差。
        ]),
    download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
)
# 批训练 50个samples， 1  channel，28x28 (50,1,28,28)
# Torch中的DataLoader是用来包装数据的工具，它能帮我们有效迭代数据，这样就可以进行批训练
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=batch_size_train,
    shuffle=True  # 是否打乱数据，一般都打乱
)
test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=batch_size_test,
    shuffle=True  # 是否打乱数据，一般都打乱
)

# 用class类来建立CNN模型
# CNN流程：卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        展平多维的卷积成的特征图->接入全连接层(Linear)->输出


class CNN(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN, self).__init__()
        # 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=1,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
                out_channels=16,  # n_filters 卷积核的高度
                kernel_size=5,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            # 激活函数
            nn.ReLU(),
            # 池化，下采样
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
            # 输出图像大小(16,14,14)
        )
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小(16,14,14)
            nn.Conv2d(  # 也可以直接简化写成nn.Conv2d(16,32,5,1,2)
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        self.conv2_drop = nn.Dropout2d()  # 正则化技术，在训练期间随机丢弃（将值设置为零）神经网络中的一些节点（或神经元），以防止过拟合。
        # 建立全卷积连接层
        self.out1 = nn.Linear(32 * 7 * 7, 50)
        self.out2 = nn.Linear(50,10)  # 输出是10个类

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batch_size
        x = self.conv2_drop(x)
        x = x.view(x.size(0), -1)  # 将张量 x 进行形状变换
        x = self.out1(x)
        x = self.out2(x)
        return x


# 初始化网络和优化器
cnn = CNN()
# 优化器选择Adam
optimizer = torch.optim.Adam(cnn.parameters(), lr=learn_rate)
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(the_epochs + 1)]
# 训练
# 把x和y 都放入Variable中，然后放入cnn中计算output，最后再计算误差


def train(EPOCH):

    cnn.train() # 和dropout层联用
    # 损失函数
    loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

                torch.save(cnn.state_dict(), 'my_model.pth')  # 保存模型


train(1)

def test():
    cnn.eval()
    test_loss =0
    correct = 0
# 加载模型，调用时需将前面训练及保存模型的代码注释掉，否则会再训练一遍
#cnn.load_state_dict(torch.load('my_model.pth'))

# print 10 predictions from test data
inputs = test_x[:30]  # 测试32个数据
test_output = cnn(inputs)
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')  # 打印识别后的数字
print(test_y[:30].numpy(), 'real number')

img = torchvision.utils.make_grid(inputs)
img = img.numpy().transpose(1, 2, 0)

# 下面三行为改变图片的亮度
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
cv2.imshow('win', img)  # opencv显示需要识别的数据图片
key_pressed = cv2.waitKey(0)
