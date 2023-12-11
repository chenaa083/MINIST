import get_model
import torch
import numpy as np
import matplotlib.pyplot as plt

# 加载已经训练好的模型
model = get_model.CNN()
model_path = 'model.pth'
model_state_dict = torch.load(model_path)
model.load_state_dict(model_state_dict)

# 测试模型
num_samples = 15
num_rows = 3
num_cols = num_samples//num_rows

model.eval()
with torch.no_grad():
    for i in range(num_samples):
        # 随机选择一个测试样本
        idx = np.random.randint(0, len(get_model.test_data))
        test_image, true_label = get_model.test_data[idx]

        # 在模型上进行前向传播
        test_image = test_image.unsqueeze(0)  # 添加 batch 维度
        output = model(test_image)
        predicted_label = torch.argmax(output).item()

        # 可视化测试结果
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(test_image.squeeze().numpy(), cmap='gray')
        plt.title(f'True: {true_label}\nPredicted: {predicted_label}')
        plt.axis('off')
plt.show()


