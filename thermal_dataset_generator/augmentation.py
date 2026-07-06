"""Image-space and sensor-space augmentations for thermal domain randomization."""

from __future__ import annotations

import cv2
import numpy as np

from .config import GeneratorConfig


class ThermalAugmentor:
    """Applies thermal sensor noise and camera perturbations."""

    def __init__(self, config: GeneratorConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.size = config.image_size

    def apply_temperature_noise(self, matrix: np.ndarray) -> np.ndarray:
        """Add gaussian noise, sensor drift, hot/dead pixels, and blur to matrix."""
        noisy = matrix.copy().astype(np.float32)
        noisy += self.rng.normal(0.0, self.config.noise_level, size=noisy.shape).astype(np.float32)

        drift = float(self.rng.uniform(*self.config.thermal_drift_range))
        noisy += drift

        h, w = noisy.shape
        if self.rng.random() < self.config.impulse_noise_probability:
            hot_count = int(h * w * self.config.hot_pixel_ratio)
            dead_count = int(h * w * self.config.dead_pixel_ratio)

            if hot_count > 0:
                ys_hot = self.rng.integers(0, h, size=hot_count)
                xs_hot = self.rng.integers(0, w, size=hot_count)
                noisy[ys_hot, xs_hot] += self.rng.uniform(4.0, 14.0, size=hot_count).astype(np.float32)

            if dead_count > 0:
                ys_dead = self.rng.integers(0, h, size=dead_count)
                xs_dead = self.rng.integers(0, w, size=dead_count)
                noisy[ys_dead, xs_dead] -= self.rng.uniform(3.0, 10.0, size=dead_count).astype(np.float32)

        sigma = max(0.1, self.config.blur_level)
        noisy = cv2.GaussianBlur(noisy, (0, 0), sigmaX=sigma, sigmaY=sigma)

        if self.rng.random() < 0.4:
            k = int(self.rng.choice([3, 5, 7]))
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            noisy = cv2.filter2D(noisy, -1, kernel)

        return noisy.astype(np.float32)

    def apply_camera_transform(
        self,
        matrix: np.ndarray,
        cpu_mask: np.ndarray,
        component_masks: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], float, tuple[int, int]]:
        """Apply random camera rotation and translation to scene outputs."""
        angle = float(self.rng.uniform(*self.config.rotation_range))
        tx = int(self.rng.integers(self.config.translation_range[0], self.config.translation_range[1] + 1))
        ty = int(self.rng.integers(self.config.translation_range[0], self.config.translation_range[1] + 1))

        m = cv2.getRotationMatrix2D((self.size * 0.5, self.size * 0.5), angle, 1.0)
        m[0, 2] += tx
        m[1, 2] += ty

        transformed = cv2.warpAffine(matrix, m, (self.size, self.size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        transformed_mask = cv2.warpAffine(cpu_mask, m, (self.size, self.size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        transformed_components: dict[str, np.ndarray] = {}
        for name, mask in component_masks.items():
            transformed_components[name] = cv2.warpAffine(
                mask,
                m,
                (self.size, self.size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )

        return transformed.astype(np.float32), transformed_mask.astype(np.uint8), transformed_components, angle, (tx, ty)

    def apply_rgb_jitter(self, rgb: np.ndarray) -> np.ndarray:
        """Apply brightness and contrast perturbations in RGB space."""
        alpha = float(self.rng.uniform(*self.config.contrast_range))
        beta_scale = float(self.rng.uniform(*self.config.brightness_range))
        beta = int((beta_scale - 1.0) * 255.0)
        out = cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)
        return out.astype(np.uint8)
