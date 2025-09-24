import torch
import matplotlib.pyplot as plt
import os

# 设置随机种子，保证结果可复现
torch.manual_seed(42)

# ---------- 1. 定义 3D Gaussian 参数 ----------
num_gaussians = 5   # 生成多少个 3D 高斯分布
points_per_gaussian = 1000  # 每个高斯采样多少点

means = torch.randn(num_gaussians, 3) * 3
covs = torch.stack([torch.eye(3) * (0.5 + torch.rand(1)) for _ in range(num_gaussians)])

# ---------- 2. 从高斯分布中采样 ----------
all_points = []
for i in range(num_gaussians):
    dist = torch.distributions.MultivariateNormal(means[i], covs[i])
    samples = dist.sample((points_per_gaussian,))
    all_points.append(samples)

points = torch.cat(all_points, dim=0)  # (N, 3)

# ---------- 3. 投影到 2D ----------
xy_points = points[:, :2]
z_values = points[:, 2]  # 用 z 值作为强度参考

# ---------- 4. 定义 2D 图像网格 ----------
img_size = 256
grid_min, grid_max = -6, 6
grid = torch.zeros((img_size, img_size))

# 映射到像素坐标
x_idx = ((xy_points[:, 0] - grid_min) / (grid_max - grid_min) * (img_size - 1)).long()
y_idx = ((xy_points[:, 1] - grid_min) / (grid_max - grid_min) * (img_size - 1)).long()

mask = (x_idx >= 0) & (x_idx < img_size) & (y_idx >= 0) & (y_idx < img_size)
x_idx, y_idx, z_values = x_idx[mask], y_idx[mask], z_values[mask]

# ---------- 5. 累加投影 ----------
for x, y, z in zip(x_idx, y_idx, z_values):
    grid[y, x] += torch.exp(-0.5 * (z**2))  # z 越接近 0，强度越大

# ---------- 6. 保存图像 ----------
save_dir = "/mnt/d/AAA牛马大学牲/论文学习/W1"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "gaussian_projection.png")

plt.figure(figsize=(6, 6))
plt.imshow(grid.numpy(), cmap="inferno", origin="lower")
plt.colorbar(label="Intensity")
plt.title("3D Gaussians projected to 2D Image Grid")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"图像已保存到: {save_path}")
