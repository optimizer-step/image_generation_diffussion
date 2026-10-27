"""CHAPTER 2 & 5: Mess up real photos, then build images from static."""
import torch
from config import NOISE_STEPS, IMAGE_SIZE


class Diffusion:
    def __init__(self, device):
        self.device = device
        self.betas = torch.linspace(0.0001, 0.04, NOISE_STEPS, device=device)
        self.alphas = 1 - self.betas
        self.kept = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean_images, steps):
        real_noise = torch.randn_like(clean_images)
        how_much_to_keep = self.kept[steps].view(-1, 1, 1, 1)
        noisy_images = (
            how_much_to_keep.sqrt() * clean_images
            + (1 - how_much_to_keep).sqrt() * real_noise
        )
        return noisy_images, real_noise

    @torch.no_grad()
    def sample(self, model, count=4):
        model.eval()
        pictures = torch.randn(count, 3, IMAGE_SIZE, IMAGE_SIZE, device=self.device)
        snapshots = []
        moment = {NOISE_STEPS - 1, 3 * NOISE_STEPS // 4, NOISE_STEPS // 2, NOISE_STEPS // 4, 0}
        for step in reversed(range(NOISE_STEPS)):
            time = torch.full((count,), step, dtype=torch.long, device=self.device)
            predicted_noise = model(pictures, time)
            beta = self.betas[step]
            alpha = self.alphas[step]
            kept = self.kept[step]
            mean = (pictures - beta / (1 - kept).sqrt() * predicted_noise) / alpha.sqrt()
            if step > 0:
                previous_kept = self.kept[step - 1]
                variance = beta * (1 - previous_kept) / (1 - kept)
                pictures = mean + variance.sqrt() * torch.randn_like(pictures)
            else:
                pictures = mean
            if step in moment:
                snapshots.append((step, pictures[:1].detach().float().cpu().clamp(-1, 1)))
        model.train()
        return pictures.clamp(-1, 1), snapshots
