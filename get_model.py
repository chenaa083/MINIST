# 训练+测试
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# 用GPU训练
device = torch.device("cuda:0")

# 使用随机化种子使神经网络的初始化每次都相同
torch.manual_seed(1)

# 超参数
the_epochs = 7  # 训练整批数据的次数
batch_size_train = 50  # 训练集批处理大小为50
batch_size_test = 1000  # 测试集批处理大小为1000
learn_rate = 0.01  # 学习率为0.01
momentum = 0.5
DOWNLOAD_MNIST = True  # 表示还没有下载数据集，如果数据集下载好了就写False
log_interval = 10

# 下载MNIST手写数据集
train_data = torchvision.datasets.MNIST(
    root='./data/',  # 保存或提取的位置  会放在当前文件夹中
    train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
        torchvision.transforms.Normalize((0.1301,), 0.3088),  # 设置 MNIST 数据集全局平均值和标准偏差。
    ]),
    download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
)

test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False,  # 表明是测试集
    transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
            torchvision.transforms.Normalize((0.1301,), 0.3088),  # 设置 MNIST 数据集全局平均值和标准偏差。
        ]),
    download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
)

# DataLoader用来包装数据，它能有效迭代数据，这样就可以进行批训练
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=batch_size_train,  # 批处理大小为50
    shuffle=True  # 打乱数据
)
test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=batch_size_test,  # 批处理大小为1000
    shuffle=True  # 打乱数据
)


# 用class类来建立CNN模型
# CNN流程：卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        展平多维的卷积成的特征图->接入全连接层(Linear)->输出


class CNN(nn.Module):  # 新建CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN, self).__init__()
        # 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=1,  # MNIST数据集是灰度图像只有一个通道
                out_channels=16,  # n_filters 卷积核的高度
                kernel_size=5,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=2,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            # 激活函数Relu
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
        self.out2 = nn.Linear(50, 10)  # 输出是10个类

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
        return F.log_softmax(x, dim=1)


# 初始化网络和优化器
cnn = CNN()
cnn = cnn.to(device)

# 优化器
optimizer = torch.optim.SGD(cnn.parameters(), lr=learn_rate, momentum=momentum)  # 选用SGD优化器，其中学习率为0.01，动量为0.5

# 创建3个列表用于存储损失值，其中train_counter，test_counter为迭代次数
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(the_epochs + 1)]

# 测试和训练


def train_model(epoch1):
    cnn.train()
    # 将其索引保存在 batch_idx1 中，将输入数据保存在 data 中，将对应的目标标签保存在 target 中
    for batch_idx1, (data, target) in enumerate(train_loader):
        # 将输入数据和目标数据移动到GPU上
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output1 = cnn(data)
        loss = F.nll_loss(output1, target)
        loss.backward()
        optimizer.step()
        if batch_idx1 % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch1, batch_idx1 * len(data), len(train_loader.dataset),
                100. * batch_idx1 / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx1*64) + ((epoch1-1)*len(train_loader.dataset)))
            torch.save(cnn.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


def test_model():
    cnn.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output2 = cnn(data)
            test_loss += F.nll_loss(output2, target, reduction='sum').item()
            pred = output2.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    test_model()
    for epoch2 in range(1, the_epochs + 1):
        train_model(epoch2)
        test_model()

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


