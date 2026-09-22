import torch
import torch.nn as nn
import torch.nn.functional as F


class DataAugmentation(nn.Module):
    """
    Sinh ra 2 augmented views từ dữ liệu gốc.
    Không dùng magnitude scaling theo yêu cầu.
    """

    def __init__(self, jitter_sigma=0.03, mask_ratio=0.10):
        super().__init__()
        self.jitter_sigma = jitter_sigma
        self.mask_ratio = mask_ratio

    def forward(self, x):
        # View 1: Gaussian jittering
        noise1 = torch.randn_like(x) * self.jitter_sigma
        x1 = x + noise1

        # View 2: Gaussian jittering + Temporal Masking (10%)
        noise2 = torch.randn_like(x) * self.jitter_sigma
        x2 = x + noise2

        # Masking dọc theo trục thời gian (dim=1)
        # Tạo mask ngẫu nhiên: True (giữ lại 90%), False (đưa về 0 khoảng 10%)
        mask = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > self.mask_ratio
        x2 = x2 * mask.float()

        return x1, x2


class ProjectionHead(nn.Module):
    """
    Linear(input_dim, 64) -> ReLU -> Linear(64, 32)
    """

    def __init__(self, input_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h):
        return self.net(h)


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE / NT-Xent Loss
    Positive pairs = (z1, z2) từ cùng 1 sample.
    Negative pairs = Các samples khác trong minibatch.
    """

    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2: [Batch, out_dim], đã được L2 normalized
        B = z1.size(0)

        # Gom z1 và z2 thành ma trận [2B, out_dim]
        z = torch.cat([z1, z2], dim=0)

        # Tính ma trận similarity: [2B, 2B]
        sim_matrix = torch.exp(torch.matmul(z, z.T) / self.temperature)

        # Xóa đường chéo chính (self-similarity)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * B, device=z.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * B, -1)

        # Tính similarity cho positive pairs
        pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        # InfoNCE loss
        loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1)).mean()
        return loss
