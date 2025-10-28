import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
from LSNet4 import LSNet

def preprocess_image(image_path, depth_path, size=224, device='cpu'):
    image = Image.open(image_path).convert('RGB')
    depth = Image.open(depth_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)  # 将图像移到指定设备上
    depth = transform(depth).unsqueeze(0).repeat(1, 3, 1, 1).to(device)  # 将深度图像移到指定设备上
    return image, depth

def visualize_and_save_features(features, feature_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for feature, name in zip(features, feature_names):
        feature = feature.squeeze(0)  # Reduce batch dimension
        mean_feature = torch.mean(feature, dim=0).detach().cpu().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(mean_feature, cmap='viridis')
        plt.axis('off')
        save_path = os.path.join(save_dir, f"{name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

# 初始化模型和加载权重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备
model = LSNet().to(device)  # 将模型加载到GPU（如果可用）
checkpoint_path = 'resultNet_epoch_best.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# 预处理图像
image_path = '/media/w/2T/HLP/data/UVT2000/RGB/38.jpg'  # 更新为正确的路径
depth_path = '/media/w/2T/HLP/data/UVT2000/T/38.jpg'
# image_path = '/media/w/2T/HLP/data/VT5000/Test/RGB/52.jpg'  # 更新为正确的路径
# depth_path = '/media/w/2T/HLP/data/VT5000/Test/T/52.jpg'
image, depth = preprocess_image(image_path, depth_path, device=device)

# 计算特征
outputs = model(image, depth)  # outputs 包含了所有特征层

# 特征名称列表
feature_names = ['out','A1', 'A2', 'A3', 'A4', 'A5','A6','A1_t', 'A2_t', 'A3_t', 'A4_t', 'A5_t', 'A6_t','F1','F2','F3','F4','F5','F6']
# feature_names = ['out','A1', 'A2', 'A3', 'A4', 'A5','A1_t', 'A2_t', 'A3_t', 'A4_t', 'A5_t','F1','F2','F3','F4','F5']
# feature_names = ['out','A1', 'A2', 'A3', 'A4', 'A1_t', 'A2_t', 'A3_t', 'A4_t', 'F1','F2','F3','F4']
# 可视化和保存特征
feature_save_path = './feature/UVT20000-38-Qmamba-ISM-10'
visualize_and_save_features(outputs[0:], feature_names, feature_save_path)  # outputs[0] 是 out，从 outputs[1] 开始是特征
