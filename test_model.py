import get_model
import torch
import numpy as np
# 加载已经训练好的模型
model = get_model.CNN()
model.load_state_dict(torch.load('model.pth'))  # 替换为你保存的模型文件的路径
model.eval()
# 测试模型
num_samples = 5

with torch.no_grad():
    for i in range(num_samples):
        # 随机选择一个测试样本
        idx = np.random.randint(0, len(test_dataset))
        test_image, true_label = test_dataset[idx]

        # 在模型上进行前向传播
        test_image = test_image.unsqueeze(0)  # 添加 batch 维度
        output = model(test_image)
        predicted_label = torch.argmax(output).item()

        # 可视化测试结果
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(test_image.squeeze().numpy(), cmap='gray')
        plt.title(f'True: {true_label}\nPredicted: {predicted_label}')
        plt.axis('off')

plt.show()
