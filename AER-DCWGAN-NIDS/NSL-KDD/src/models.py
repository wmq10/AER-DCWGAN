import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from config import BATCH_SIZE, EPOCHS
from config import LATENT_DIM_MIN, LATENT_DIM_MAX, RECON_WEIGHT_MIN, RECON_WEIGHT_MAX, GP_WEIGHT_MIN, GP_WEIGHT_MAX, \
    AER_WEIGHT_MIN, AER_WEIGHT_MAX
from torch.nn.utils import spectral_norm, clip_grad_norm_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 改进的编码器模块（添加权重初始化）
class Encoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(img_shape), 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


# 增强的条件生成器（含标签嵌入）
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape, num_classes):
        super().__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(num_classes, latent_dim)  # 标签嵌入层
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),  # 噪声+标签嵌入
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 80),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(0.2),
            nn.Linear(80, np.prod(img_shape)),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = torch.cat([z, self.label_emb(labels)], dim=1)  # 拼接噪声与标签嵌入
        return self.net(z).view(-1, *self.img_shape)


# 条件判别器（含标签拼接）
class ConditionalDiscriminator(nn.Module):
    def __init__(self, latent_dim, img_shape, num_classes):
        super().__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(num_classes, latent_dim)  # 标签嵌入层
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim + np.prod(img_shape) + latent_dim, 100)),  # 噪声+图像+标签
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(100, 80)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(80, 60)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(60, 40)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(40, 20)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(20, 1)),
        )

    def forward(self, z, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_emb = self.label_emb(labels)
        combined = torch.cat([z, img_flat, label_emb], dim=1)  # 三输入拼接
        return self.net(combined)


# 编码器判别器（对抗编码器正则化）
class EncoderDiscriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 40),
            nn.LeakyReLU(0.2),
            nn.Linear(40, 60),
            nn.LeakyReLU(0.2),
            nn.Linear(60, 80),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(80),  # 添加BatchNorm稳定训练
            nn.Linear(80, 60),
            nn.LeakyReLU(0.2),
            nn.Linear(60, 40),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # 添加Dropout防止过拟合
            nn.Linear(40, 20),
            nn.LeakyReLU(0.2),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)


# 梯度惩罚计算（含标签条件）
# 梯度惩罚计算（含标签条件）
def compute_gradient_penalty(model, real_imgs, fake_imgs, noise, labels):
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=False)
    interpolates_img = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    d_interpolates = model.discriminator(noise.detach(), interpolates_img, labels)  # 使用判别器
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates_img,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return model.gp_weight * gradient_penalty  # 从模型实例获取 gp_weight


class AE_WGAN_AER(nn.Module):
    def __init__(self, img_shape, num_classes, latent_dim=32, recon_weight=1.5, gp_weight=3, aer_weight=2):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.recon_weight = recon_weight
        self.gp_weight = gp_weight
        self.aer_weight = aer_weight
        self.num_classes = num_classes  # 类别数

        # 初始化条件模块
        self.encoder = Encoder(img_shape, latent_dim).to(device)
        self.generator = ConditionalGenerator(latent_dim, img_shape, num_classes).to(device)
        self.discriminator = ConditionalDiscriminator(latent_dim, img_shape, num_classes).to(device)
        self.encoder_discriminator = EncoderDiscriminator(latent_dim).to(device)

        # 优化器
        self.optimizer_E = optim.Adam(self.encoder.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=5e-5, betas=(0.5, 0.9))


        self.encoder_discriminator = EncoderDiscriminator(latent_dim).to(device)
        self.optimizer_ED = optim.Adam(self.encoder_discriminator.parameters(), lr=5e-5, betas=(0.5, 0.9))
        self.bce_loss = nn.BCELoss()  # 新增：二元交叉熵损失

    def train(self, attack_name, reshaped_X_attack, labels, epochs=EPOCHS, batch_size=BATCH_SIZE):
        self.to(device)
        dataset = AE_WGAN_Dataset(reshaped_X_attack, labels, self.img_shape)

        # 类别权重采样
        unique_labels = np.unique(labels)
        class_sample_count = np.array([(labels == label).sum() for label in unique_labels])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[np.where(unique_labels == label)[0][0]] for label in labels])
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        metrics = {
            'epoch': [],
            'd_loss': [],
            'g_loss': [],
            'wasserstein': [],
            'diversity': [],
            'd_accuracy': [],
            'ed_loss': [],
            'recon_loss': []
        }

        print("开始联合训练...")
        for epoch in range(epochs):
            # 初始化累计变量（修正此处）
            d_correct = 0
            d_total = 0
            epoch_recon_loss = 0.0
            epoch_ed_loss = 0.0
            D_real_total = 0.0  # 累计真实样本评分总和
            D_fake_total = 0.0  # 累计生成样本评分总和

            for i, (real_imgs, real_labels) in enumerate(dataloader):
                real_imgs = real_imgs.to(device)
                real_labels = real_labels.to(device)
                batch_size_curr = real_imgs.size(0)

                # 训练判别器 5 次
                for _ in range(5):
                    self.optimizer_D.zero_grad()
                    noise = torch.randn(batch_size_curr, self.latent_dim, device=device)

                    with torch.no_grad():
                        fake_imgs = self.generator(noise, real_labels)

                    gp = compute_gradient_penalty(self, real_imgs, fake_imgs, noise, real_labels)

                    # 计算判别器对真实和生成样本的评分
                    D_real_out = self.discriminator(self.encoder(real_imgs).detach(), real_imgs, real_labels)
                    D_fake_out = self.discriminator(noise.detach(), fake_imgs.detach(), real_labels)

                    # 累计评分总和（修正此处）
                    D_real_total += D_real_out.sum().item()
                    D_fake_total += D_fake_out.sum().item()

                    d_loss = -D_real_out.mean() + D_fake_out.mean() + self.gp_weight * gp
                    d_loss.backward()
                    clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                    self.optimizer_D.step()

                    # 累计准确率统计
                    d_correct += ((D_real_out > 0).sum() + (D_fake_out < 0).sum()).item()
                    d_total += batch_size_curr * 2

                # 训练编码器判别器（ED）
                self.optimizer_ED.zero_grad()
                z_enc_real = self.encoder(real_imgs)
                ed_real = self.encoder_discriminator(z_enc_real)
                loss_ed_real = self.bce_loss(ed_real, torch.ones_like(ed_real))

                z_enc_fake = torch.randn(batch_size_curr, self.latent_dim, device=device)
                ed_fake = self.encoder_discriminator(z_enc_fake)
                loss_ed_fake = self.bce_loss(ed_fake, torch.zeros_like(ed_fake))
                ed_loss = (loss_ed_real + loss_ed_fake) / 2
                ed_loss.backward()
                self.optimizer_ED.step()
                epoch_ed_loss += ed_loss.item()

                # 训练生成器和编码器
                self.optimizer_G.zero_grad()
                self.optimizer_E.zero_grad()
                z = torch.randn(batch_size_curr, self.latent_dim, device=device)
                gen_imgs = self.generator(z, real_labels)
                D_gen = self.discriminator(z, gen_imgs, real_labels)
                recon_imgs = self.generator(self.encoder(real_imgs), real_labels)
                recon_loss = nn.MSELoss()(recon_imgs, real_imgs)
                z_enc = self.encoder(real_imgs)
                ED_enc = self.encoder_discriminator(z_enc)
                aer_loss = -ED_enc.mean()
                g_loss = -D_gen.mean() + self.recon_weight * recon_loss + self.aer_weight * aer_loss
                g_loss.backward()
                clip_grad_norm_([*self.generator.parameters(), *self.encoder.parameters()], max_norm=2.0)
                self.optimizer_G.step()
                self.optimizer_E.step()

                epoch_recon_loss += recon_loss.item()

            # 计算指标（修正Wasserstein距离计算）
            with torch.no_grad():
                z_eval = torch.randn(1000, self.latent_dim, device=device)
                labels_eval = torch.randint(0, self.num_classes, (1000,), device=device)
                gen_eval = self.generator(z_eval, labels_eval)
                diversity = gen_eval.cpu().std(dim=0).mean().item()

                # 计算平均Wasserstein距离（真实均值 - 生成均值）
                if d_total > 0:
                    real_mean = D_real_total / d_total  # 真实样本评分均值
                    fake_mean = D_fake_total / d_total  # 生成样本评分均值
                    wasserstein = real_mean - fake_mean  # 正确计算方式
                else:
                    wasserstein = 0.0

                d_accuracy = d_correct / d_total if d_total != 0 else 0.0
                current_recon_loss = epoch_recon_loss / len(dataloader)
                current_ed_loss = epoch_ed_loss / len(dataloader)

            # 记录指标
            metrics['epoch'].append(epoch)
            metrics['d_loss'].append(d_loss.item() if 'd_loss' in locals() else 0.0)
            metrics['g_loss'].append(g_loss.item() if 'g_loss' in locals() else 0.0)
            metrics['wasserstein'].append(wasserstein)
            metrics['diversity'].append(diversity)
            metrics['d_accuracy'].append(d_accuracy)
            metrics['recon_loss'].append(current_recon_loss)
            metrics['ed_loss'].append(current_ed_loss)

            print(
                f"Epoch {epoch:03d} | D Loss: {d_loss.item():.2f} | G Loss: {g_loss:.2f} | "
                f"ED Loss: {current_ed_loss:.4f} | W-Dist: {wasserstein:.2f} | Div: {diversity:.2f} | "
                f"Recon Loss: {current_recon_loss:.4f}"
            )

            if epoch % 10 == 0:
                self._visualize_progress(metrics, attack_name)

        return self.generate_data(num_samples=len(reshaped_X_attack))

    def _visualize_progress(self, metrics, attack_name):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(metrics['epoch'], metrics['d_loss'], label='D Loss')
        plt.plot(metrics['epoch'], metrics['g_loss'], label='G Loss')
        plt.plot(metrics['epoch'], metrics['recon_loss'], label='Recon Loss')
        plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(metrics['epoch'], metrics['wasserstein'], 'r-', label='W-Dist')
        plt.xlabel('Epoch'), plt.ylabel('Wasserstein Distance'), plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(metrics['epoch'], metrics['diversity'], 'm-', label='Diversity')
        plt.xlabel('Epoch'), plt.ylabel('Feature Diversity'), plt.legend()

        plt.suptitle(f'Training Metrics - {attack_name}')
        plt.tight_layout()
        plot_dir = os.path.join(".", "generated", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'AE_WGAN_AER_{attack_name}.png'))
        plt.close()

    def generate_data(self, num_samples, target_label=None):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            if target_label is not None:
                labels = torch.full((num_samples,), target_label, dtype=torch.long, device=device)
            else:
                labels = torch.randint(0, self.num_classes, (num_samples,), device=device)  # 随机生成标签
            generated = self.generator(z, labels).cpu().numpy()
        return generated.reshape(-1, np.prod(self.img_shape))


class AE_WGAN_Dataset(Dataset):
    def __init__(self, data, labels, img_shape):
        self.img_shape = img_shape
        self.data = torch.tensor(data, dtype=torch.float32).view(-1, *img_shape)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 使用示例
if __name__ == "__main__":
    # 模拟数据（1000样本，特征维度需匹配img_shape=(1,4,6)即24维）
    mock_data = np.random.randn(1000, 24).astype(np.float32)
    mock_labels = np.random.randint(0, 5, 1000)  # 5个类别

    # 初始化模型（指定类别数为5）
    model = AE_WGAN_AER(
        img_shape=(1, 4, 6),
        num_classes=5,
        latent_dim=20,
        recon_weight=3,
        gp_weight=5,
        aer_weight=5
    )

    generated = model.train(
        attack_name="TestAttack",
        reshaped_X_attack=mock_data,
        labels=mock_labels,
        epochs=200,
        batch_size=64
    )

    pd.DataFrame(generated).to_csv("generated_samples_AER.csv", index=False)