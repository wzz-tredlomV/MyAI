#!/usr/bin/env python3
"""
训练脚本
训练风格编码器和生成器
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm

from src.vector_style_encoder import HandwritingStyleEncoder
from src.vector_generator import VectorHandwritingGenerator


class HandwritingDataset(Dataset):
    """
    手写数据集
    需要成对数据：(同一个人的多个字迹样本, 对应的标准字体)
    """
    def __init__(self, 
                 data_dir: str,
                 transform=None,
                 max_samples: int = None):
        self.data_dir = Path(data_dir)
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # 加载数据索引
        self.samples = self._load_index()
        if max_samples:
            self.samples = self.samples[:max_samples]

    def _load_index(self) -> List[Dict]:
        """加载数据索引"""
        index_file = self.data_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)

        # 自动扫描目录结构
        samples = []
        for writer_dir in self.data_dir.glob("writer_*"):
            writer_id = writer_dir.name
            images = list(writer_dir.glob("*.png")) + list(writer_dir.glob("*.jpg"))

            for img_path in images:
                # 假设对应的标准字体在同名的 skeleton 目录中
                skeleton_path = self.data_dir / "skeletons" / img_path.name
                if skeleton_path.exists():
                    samples.append({
                        'writer_id': writer_id,
                        'handwriting': str(img_path),
                        'skeleton': str(skeleton_path)
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载手写样本
        hw_img = Image.open(sample['handwriting']).convert('L')
        hw_tensor = self.transform(hw_img)

        # 加载骨架图
        sk_img = Image.open(sample['skeleton']).convert('L')
        sk_tensor = self.transform(sk_img)

        return {
            'handwriting': hw_tensor,
            'skeleton': sk_tensor,
            'writer_id': sample['writer_id']
        }


class StyleConsistencyLoss(nn.Module):
    """风格一致性损失"""
    def __init__(self):
        super().__init__()

    def forward(self, style1, style2):
        return torch.mean((style1 - style2) ** 2)


class StrokeSmoothnessLoss(nn.Module):
    """笔画平滑度损失"""
    def __init__(self):
        super().__init__()

    def forward(self, stroke_params):
        # 鼓励笔画参数变化平滑
        diff = stroke_params[:, 1:, :] - stroke_params[:, :-1, :]
        return torch.mean(diff ** 2)


def train_epoch(encoder, generator, discriminator, 
                dataloader, optimizers, device, epoch):
    """训练一个epoch"""
    encoder.train()
    generator.train()
    discriminator.train()

    opt_enc, opt_gen, opt_disc = optimizers

    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        hw = batch['handwriting'].to(device)
        skeleton = batch['skeleton'].to(device)

        # 提取风格
        style_features = encoder(hw)
        global_style = style_features['global']

        # 生成
        generated = generator(skeleton, global_style)

        # 重建损失
        recon_loss = nn.L1Loss()(generated['presence_map'], 
                                 (hw > 0).float().unsqueeze(1))

        # 风格一致性
        style_loss = StyleConsistencyLoss()(global_style, 
                                            encoder(hw)['global'])

        # 笔画平滑
        smooth_loss = StrokeSmoothnessLoss()(generated['stroke_params'])

        # 总损失
        loss = recon_loss + 0.1 * style_loss + 0.01 * smooth_loss

        # 反向传播
        opt_enc.zero_grad()
        opt_gen.zero_grad()
        loss.backward()
        opt_enc.step()
        opt_gen.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='数据集目录')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    encoder = HandwritingStyleEncoder().to(device)
    generator = VectorHandwritingGenerator().to(device)

    # 简化版判别器
    discriminator = nn.Sequential(
        nn.Conv2d(1, 64, 4, 2, 1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(64, 128, 4, 2, 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),
        nn.Conv2d(128, 1, 4, 1, 1),
        nn.Sigmoid()
    ).to(device)

    # 优化器
    opt_enc = optim.Adam(encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # 数据加载
    dataset = HandwritingDataset(args.data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=True, num_workers=4)

    # 训练循环
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(encoder, generator, discriminator,
                              dataloader, (opt_enc, opt_gen, opt_disc),
                              device, epoch)

        print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}")

        # 保存检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
            }, save_dir / f"checkpoint_epoch_{epoch}.pth")

    # 保存最终模型
    torch.save(encoder.state_dict(), save_dir / "style_encoder_final.pth")
    torch.save(generator.state_dict(), save_dir / "generator_final.pth")
    print("训练完成！")


if __name__ == '__main__':
    main()
