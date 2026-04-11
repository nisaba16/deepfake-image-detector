"""
ForensicMobileNetV3 — MobileNetV3-Small with configurable multi-channel forensic input.

The model accepts a standard ImageNet-normalized RGB tensor and internally computes
any combination of the following feature channels before the backbone:

  rgb   (3 ch)  Standard ImageNet-normalized RGB — kept normalized for the backbone.
  hsv   (3 ch)  Hue / Saturation / Value — color manipulation cues.
  fft   (3 ch)  Log-magnitude 2-D FFT per channel — GAN artifacts appear as
                periodic patterns in the frequency domain.
  noise (1 ch)  High-frequency noise residual (grayscale) — real cameras have a
                sensor noise fingerprint; generators do not.
  srm   (3 ch)  Three SRM (Steganalysis Rich Model) high-pass kernels — widely used
                in image-forgery detection to expose editing traces.

Total channels per feature combination:
  rgb only          →  3 ch
  rgb + hsv         →  6 ch
  rgb + hsv + fft   →  9 ch
  rgb + hsv + noise → 7 ch
  all five          → 13 ch

The first Conv2d of MobileNetV3-Small is replaced with an N-channel equivalent.
When pretrained_rgb=True and 'rgb' is in features, the RGB slice of the new conv
is warm-initialised from ImageNet pretrained weights so training starts from a
sensible representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

AVAILABLE_FEATURES = ("rgb", "hsv", "fft", "noise", "srm")
FEATURE_CHANNELS   = {"rgb": 3, "hsv": 3, "fft": 3, "noise": 1, "srm": 3}


# ---------------------------------------------------------------------------
# SRM kernel bank (fixed, non-trainable)
# ---------------------------------------------------------------------------
def _make_srm_kernels() -> torch.Tensor:
    """
    Three 5×5 high-pass SRM kernels from
    'Rich Models for Steganalysis of Digital Images' (Fridrich & Kodovsky, 2012).
    Returns shape (3, 1, 5, 5).
    """
    # Laplacian-style (isotropic second-order)
    k1 = torch.tensor([
        [ 0,  0,  0,  0,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  2, -4,  2,  0],
        [ 0, -1,  2, -1,  0],
        [ 0,  0,  0,  0,  0],
    ], dtype=torch.float32) / 4.0

    # Second-order (captures larger neighbourhood)
    k2 = torch.tensor([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8,-12,  8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1],
    ], dtype=torch.float32) / 12.0

    # Simple horizontal edge (asymmetric, 3-tap)
    k3 = torch.tensor([
        [0,  0,  0,  0, 0],
        [0,  0,  0,  0, 0],
        [0,  1, -2,  1, 0],
        [0,  0,  0,  0, 0],
        [0,  0,  0,  0, 0],
    ], dtype=torch.float32) / 2.0

    return torch.stack([k1, k2, k3], dim=0).unsqueeze(1)   # (3, 1, 5, 5)


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------
class ForensicFeatureExtractor(nn.Module):
    """
    Differentiable, GPU-safe multi-channel feature extractor.

    Input : (B, 3, H, W) — ImageNet-normalized float32.
    Output: (B, C, H, W) — stacked forensic features where C = sum of chosen channels.
    """

    def __init__(self, features: tuple = ("rgb", "hsv", "fft", "noise")):
        super().__init__()
        unknown = set(features) - set(AVAILABLE_FEATURES)
        if unknown:
            raise ValueError(f"Unknown features: {unknown}. Choose from {AVAILABLE_FEATURES}")

        self.features    = list(features)
        self.out_channels = sum(FEATURE_CHANNELS[f] for f in self.features)

        # ImageNet stats as buffers so they move with .to(device)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std",  std)

        if "srm" in self.features:
            self.register_buffer("_srm_kernels", _make_srm_kernels())

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------
    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalized → [0, 1] float32."""
        return (x * self._std + self._mean).clamp(0.0, 1.0)

    @staticmethod
    def _rgb_to_hsv(x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W) in [0, 1]
        Returns (B, 3, H, W) HSV in [0, 1].
        """
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        max_c, _ = x.max(dim=1)
        min_c, _ = x.min(dim=1)
        delta     = (max_c - min_c).clamp(min=1e-8)

        v = max_c
        s = torch.where(max_c > 1e-8, delta / max_c.clamp(min=1e-8), torch.zeros_like(max_c))

        # Hue — computed per dominant channel
        h = torch.zeros_like(max_c)
        mask_r = max_c == r
        mask_g = (max_c == g) & ~mask_r
        mask_b = ~mask_r & ~mask_g

        h = torch.where(mask_r, ((g - b) / delta) % 6,   h)
        h = torch.where(mask_g,  (b - r) / delta + 2.0,  h)
        h = torch.where(mask_b,  (r - g) / delta + 4.0,  h)
        h = (h / 6.0).clamp(0.0, 1.0)

        return torch.stack([h, s, v], dim=1)

    @staticmethod
    def _fft_features(x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W) in [0, 1]
        Returns (B, 3, H, W) — log-magnitude 2-D FFT, normalised to [0, 1].

        GAN generators produce characteristic spectral fingerprints (regular grid
        artefacts) that are invisible in the spatial domain but obvious in the FFT.
        """
        # rfft2 output: (B, 3, H, W//2+1) complex
        fft = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(fft)           # real magnitude
        mag = torch.log1p(mag)         # log scale

        # Pad width back to W so spatial dimensions are preserved
        pad = x.shape[-1] - mag.shape[-1]
        if pad > 0:
            mag = F.pad(mag, (0, pad))

        # Per-image, per-channel normalisation to [0, 1]
        B, C, H, W = mag.shape
        mn = mag.view(B, C, -1).min(-1).values.view(B, C, 1, 1)
        mx = mag.view(B, C, -1).max(-1).values.view(B, C, 1, 1)
        return (mag - mn) / (mx - mn + 1e-8)

    @staticmethod
    def _noise_residual(x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W) in [0, 1]
        Returns (B, 1, H, W) in [0, 1] — high-frequency grayscale residual.

        Approximates the Photo Response Non-Uniformity (PRNU) noise fingerprint:
        real images have consistent sensor noise; AI-generated images do not.
        """
        gray    = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        blurred = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        # Shift to [0, 1] — residual is roughly zero-mean
        residual = (gray - blurred + 0.5).clamp(0.0, 1.0)
        return residual

    def _srm_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W) in [0, 1]
        Returns (B, 3, H, W) in [0, 1] — three SRM high-pass filter responses.

        Exposes image manipulation traces (splicing, in-painting, GAN synthesis)
        by amplifying the residual noise that editing operations disrupt.
        """
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]   # (B,1,H,W)
        out  = F.conv2d(gray, self._srm_kernels, padding=2)                    # (B,3,H,W)

        # Per-image, per-channel normalisation to [0, 1]
        B, C, H, W = out.shape
        mn = out.view(B, C, -1).min(-1).values.view(B, C, 1, 1)
        mx = out.view(B, C, -1).max(-1).values.view(B, C, 1, 1)
        return (out - mn) / (mx - mn + 1e-8)

    # -----------------------------------------------------------------------
    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        x_norm : (B, 3, H, W) — ImageNet-normalised RGB.
        Returns  (B, C, H, W) — stacked forensic features.
        """
        x = self._denorm(x_norm)          # [0, 1] for all non-RGB features
        parts = []
        for f in self.features:
            if   f == "rgb":   parts.append(x_norm)               # stays normalised
            elif f == "hsv":   parts.append(self._rgb_to_hsv(x))
            elif f == "fft":   parts.append(self._fft_features(x))
            elif f == "noise": parts.append(self._noise_residual(x))
            elif f == "srm":   parts.append(self._srm_features(x))
        return torch.cat(parts, dim=1)     # (B, out_channels, H, W)


# ---------------------------------------------------------------------------
# Training-time noise augmentation (active only during model.train())
# ---------------------------------------------------------------------------
class ForensicNoiseAugment(nn.Module):
    """
    Differentiable noise augmentations applied to the raw RGB input tensor
    during training only (no-op at eval/inference time).

    Applied in the pixel domain [0,1] before forensic feature extraction,
    which means they affect ALL feature streams (FFT, SRM, noise residual, …)
    simultaneously — the way real-world post-processing would.

    Augmentations (each applied independently with probability p):
      gaussian  — additive Gaussian noise: simulates sensor noise / unseen cameras
      jpeg      — differentiable JPEG approximation: covers compression artefacts
                  common in social-media deepfakes
      blur      — random Gaussian blur: covers slight focus differences / upsampling
      erasing   — random rectangular patch set to mean: forces spatial robustness
    """

    def __init__(
        self,
        gaussian_std: float = 0.02,   # σ for additive Gaussian noise (in [0,1] scale)
        jpeg_quality_min: int = 50,   # lowest JPEG quality to simulate
        jpeg_quality_max: int = 95,   # highest JPEG quality to simulate
        blur_kernel_max: int = 5,     # max Gaussian blur kernel radius (odd sizes 1..k)
        erase_ratio: float = 0.1,     # max fraction of image area to erase
        p_gaussian: float = 0.5,
        p_jpeg: float = 0.5,
        p_blur: float = 0.3,
        p_erase: float = 0.3,
    ):
        super().__init__()
        self.gaussian_std      = gaussian_std
        self.jpeg_quality_min  = jpeg_quality_min
        self.jpeg_quality_max  = jpeg_quality_max
        self.blur_kernel_max   = blur_kernel_max
        self.erase_ratio       = erase_ratio
        self.p_gaussian        = p_gaussian
        self.p_jpeg            = p_jpeg
        self.p_blur            = p_blur
        self.p_erase           = p_erase

    # ------------------------------------------------------------------
    def _gaussian(self, x: torch.Tensor) -> torch.Tensor:
        return (x + torch.randn_like(x) * self.gaussian_std).clamp(0.0, 1.0)

    def _jpeg(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable JPEG approximation via DCT-domain quantisation.
        Operates on grayscale luma only (fast); chroma degradation is small.
        Quality q ∈ [jpeg_quality_min, jpeg_quality_max] sampled per batch.
        """
        q = torch.randint(self.jpeg_quality_min, self.jpeg_quality_max + 1, ()).item()
        # JPEG quantisation step ≈ (100 - q) / 50 * 0.1 mapped to [0,1] noise
        qstep = (100 - q) / 5000.0
        noise = (torch.rand_like(x) - 0.5) * qstep
        return (x + noise).clamp(0.0, 1.0)

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        # Pick a random odd kernel size: 1, 3, or 5 (up to blur_kernel_max)
        max_k = max(1, self.blur_kernel_max)
        sizes = [k for k in [1, 3, 5, 7] if k <= max_k]
        k = sizes[torch.randint(len(sizes), ()).item()]
        if k == 1:
            return x
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        # Build separable Gaussian kernel
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        kx = g.view(1, 1, 1, k).expand(3, 1, 1, k)
        ky = g.view(1, 1, k, 1).expand(3, 1, k, 1)
        pad = k // 2
        out = F.conv2d(x, kx, padding=(0, pad), groups=3)
        out = F.conv2d(out, ky, padding=(pad, 0), groups=3)
        return out.clamp(0.0, 1.0)

    def _erase(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        area = H * W * self.erase_ratio
        eh = max(1, int((area ** 0.5)))
        ew = max(1, int((area ** 0.5)))
        y0 = torch.randint(0, max(1, H - eh), ()).item()
        x0 = torch.randint(0, max(1, W - ew), ()).item()
        out = x.clone()
        out[:, :, y0:y0 + eh, x0:x0 + ew] = 0.5   # set to neutral grey
        return out

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 3, H, W) in [0, 1]. Returns augmented tensor, same shape."""
        if not self.training:
            return x
        if torch.rand(()).item() < self.p_gaussian:
            x = self._gaussian(x)
        if torch.rand(()).item() < self.p_jpeg:
            x = self._jpeg(x)
        if torch.rand(()).item() < self.p_blur:
            x = self._blur(x)
        if torch.rand(()).item() < self.p_erase:
            x = self._erase(x)
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ForensicMobileNetV3(nn.Module):
    """
    MobileNetV3-Small with configurable multi-channel forensic input
    and optional training-time noise augmentation.

    Architecture:
        input (B,3,H,W)  — ImageNet-normalised RGB
            │
        ForensicNoiseAugment  ←  training only (gaussian / jpeg / blur / erase)
            │
        ForensicFeatureExtractor  ←  differentiable, GPU-safe
            │  (B, N, H, W)
            │
        Modified MobileNetV3-Small  ←  first Conv2d accepts N channels
            │
        Linear(hidden, num_classes)
    """

    def __init__(
        self,
        features: tuple = ("rgb", "hsv", "fft", "noise"),
        num_classes: int = 2,
        pretrained_rgb: bool = True,
        noise_augment: bool = True,
        noise_kwargs: dict = None,
    ):
        super().__init__()
        self.feature_names = list(features)
        self.augment       = ForensicNoiseAugment(**(noise_kwargs or {})) if noise_augment else None
        self.extractor     = ForensicFeatureExtractor(features)
        in_channels        = self.extractor.out_channels

        # ---- Load backbone ------------------------------------------------
        weights = (
            models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if pretrained_rgb else None
        )
        backbone = models.mobilenet_v3_small(weights=weights)

        # ---- Adapt first Conv2d to N input channels -----------------------
        old_conv = backbone.features[0][0]           # Conv2d(3, 16, 3, 2, 1, bias=False)
        new_conv  = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size = old_conv.kernel_size,
            stride      = old_conv.stride,
            padding     = old_conv.padding,
            bias        = old_conv.bias is not None,
        )

        if pretrained_rgb and "rgb" in features:
            # Warm-init: zero all weights, copy pretrained weights for RGB slice
            rgb_offset = self.feature_names.index("rgb") * 3
            with torch.no_grad():
                new_conv.weight.zero_()
                new_conv.weight[:, rgb_offset : rgb_offset + 3] = old_conv.weight
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        backbone.features[0][0] = new_conv

        # ---- Replace classifier head for binary task ----------------------
        in_feats = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_feats, num_classes)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 3, H, W) ImageNet-normalised RGB."""
        if self.augment is not None:
            # augment in pixel domain, then re-normalise for the RGB stream
            x_01 = (x * torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
                      + torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)).clamp(0, 1)
            x_01 = self.augment(x_01)
            x    = (x_01 - torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)) \
                        / torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        feats = self.extractor(x)       # (B, N, H, W)
        return self.backbone(feats)

    @property
    def n_input_channels(self) -> int:
        return self.extractor.out_channels

    def describe(self) -> str:
        lines = [
            "ForensicMobileNetV3",
            f"  Features    : {self.feature_names}",
            f"  In-channels : {self.n_input_channels}",
            f"  Noise augment: {self.augment is not None}",
        ]
        param_M = sum(p.numel() for p in self.parameters()) / 1e6
        lines.append(f"  Parameters  : {param_M:.2f} M")
        return "\n".join(lines)
