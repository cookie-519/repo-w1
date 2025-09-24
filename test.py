import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os

# ====== 1. 定义 Gaussian Kernel ======
def gaussian_kernel(size: int, sigma: float):
    """生成一个二维高斯核"""
    coords = torch.arange(size).float() - size // 2
    x_grid, y_grid = torch.meshgrid(coords, coords, indexing="xy")
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        kernel = gaussian_kernel(kernel_size, sigma)
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape [1,1,H,W]
        self.conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv.weight.data = self.kernel

    def forward(self, x):
        return self.conv(x)


# ====== 2. 加载图片 ======
img_path = r"/mnt/d/有用的小东西/好看的图/20200226230743_dCR8T.jpg"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"找不到文件: {img_path}")

img = Image.open(img_path).convert("L")  # 转为灰度图
transform = T.ToTensor()
x = transform(img).unsqueeze(0)  # shape [1,1,H,W]

# ====== 3. 应用高斯模糊 ======
blur = GaussianBlur(kernel_size=15, sigma=3.0)
y = blur(x)

# ====== 4. 可视化结果 ======
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(x.squeeze().numpy(), cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Gaussian Blur")
plt.imshow(y.squeeze().detach().numpy(), cmap="gray")

plt.show()
