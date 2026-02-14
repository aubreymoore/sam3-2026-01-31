import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.spatial.distance import euclidean

import os
import json
import pandas as pd
from pathlib import Path
import sqlite3
from icecream import ic
import glob


def clean_contour(contour: np.ndarray, sigma:float=1.0) -> np.ndarray:
    """Internal: Removes spikes and jitter."""
    x, y = contour[:, 0], contour[:, 1]
    x = median_filter(x, size=3, mode='wrap')
    y = median_filter(y, size=3, mode='wrap')
    x_smooth = gaussian_filter1d(x, sigma, mode='wrap')
    y_smooth = gaussian_filter1d(y, sigma, mode='wrap')
    return np.column_stack([x_smooth, y_smooth])


def calculate_efd(contour:np.ndarray, harmonics:int=20) -> np.ndarray:
    """Internal: Standard Kuhl and Giardina EFD math."""

    # Ensure the contour is closed
    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    dxy = np.diff(contour, axis=0)
    dt = np.sqrt((dxy**2).sum(axis=1))
    t = np.concatenate([[0], np.cumsum(dt)])
    T = t[-1]

    coeffs = np.zeros((harmonics, 4))
    for n in range(1, harmonics + 1):
        term = 2 * np.pi * n / T
        factor = T / (2 * (n * np.pi)**2)

        an = factor * np.sum((dxy[:, 0] / dt) * (np.cos(term * t[1:]) - np.cos(term * t[:-1])))
        bn = factor * np.sum((dxy[:, 0] / dt) * (np.sin(term * t[1:]) - np.sin(term * t[:-1])))
        cn = factor * np.sum((dxy[:, 1] / dt) * (np.cos(term * t[1:]) - np.cos(term * t[:-1])))
        dn = factor * np.sum((dxy[:, 1] / dt) * (np.sin(term * t[1:]) - np.sin(term * t[:-1])))
        coeffs[n-1] = [an, bn, cn, dn]
    return coeffs


def get_feature_vector(contour:np.ndarray) -> np.ndarray:
    """Processes a polygon and returns a normalized 1D feature vector."""
    # cleaned = clean_contour(contour)
    # coeffs = calculate_efd(cleaned)
    coeffs = calculate_efd(contour, 50)
    return normalize(coeffs)


def normalize(coeffs:np.ndarray) -> np.ndarray:
    """Internal: Rotation, size, and starting-point invariance."""
    a1, b1, c1, d1 = coeffs[0]
    theta_1 = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), (a1**2 + c1**2 - b1**2 - d1**2))

    a1_p = a1 * np.cos(theta_1) + b1 * np.sin(theta_1)
    c1_p = c1 * np.cos(theta_1) + d1 * np.sin(theta_1)
    psi_1 = np.arctan2(c1_p, a1_p)
    E = np.sqrt(a1_p**2 + c1_p**2)

    norm_v = []
    for n in range(len(coeffs)):
        an, bn, cn, dn = coeffs[n]
        cos_nt = np.cos((n + 1) * theta_1)
        sin_nt = np.sin((n + 1) * theta_1)
        cp, sp = np.cos(psi_1), np.sin(psi_1)

        # Standard matrix transformations for normalization
        an_n = (1/E) * (cp * (an*cos_nt + bn*sin_nt) + sp * (cn*cos_nt + dn*sin_nt))
        bn_n = (1/E) * (cp * (-an*sin_nt + bn*cos_nt) + sp * (-cn*sin_nt + dn*cos_nt))
        cn_n = (1/E) * (-sp * (an*cos_nt + bn*sin_nt) + cp * (cn*cos_nt + dn*sin_nt))
        dn_n = (1/E) * (-sp * (-an*sin_nt + bn*cos_nt) + cp * (-cn*sin_nt + dn*cos_nt))
        norm_v.extend([an_n, bn_n, cn_n, dn_n])
    return np.array(norm_v)


def reconstruct(feature_vector:np.ndarray, num_points:int=200) -> tuple[np.ndarray, np.ndarray] :
    """Reconstructs (x, y) coordinates from a feature vector."""
    coeffs = feature_vector.reshape(-1, 4)
    t = np.linspace(0, 2 * np.pi, num_points)
    x, y = np.zeros(num_points), np.zeros(num_points)
    for n, (a, b, c, d) in enumerate(coeffs):
        h = n + 1
        x += a * np.cos(h * t) + b * np.sin(h * t)
        y += c * np.cos(h * t) + d * np.sin(h * t)
    return x, y

# Usage example

# # Initialize your analyzer
# analyzer = ShapeDescriptor(harmonics=15, smoothing_sigma=1.2)

# # Process two different shapes
# feat1 = analyzer.get_feature_vector(polygon_data_1)
# feat2 = analyzer.get_feature_vector(polygon_data_2)

# # Compare them
# dist = euclidean(feat1, feat2)
# print(f"Morphological distance: {dist:.4f}")

# # Visualize the normalized reconstruction
# x, y = analyzer.reconstruct(feat1)
# plt.plot(x, y)
# plt.axis('equal')
# plt.show()


def visualize_harmonics(contour: np.ndarray, harmonic_list: list[int]=[1, 3, 10, 50]) -> None:
    """Plots original vs reconstructed shapes for various harmonics."""
    plt.figure(figsize=(15, 5))

    # Plot Original
    plt.subplot(1, len(harmonic_list) + 1, 1)
    plt.plot(contour[:, 0], contour[:, 1], 'k--', alpha=0.5)
    plt.title("Original Boundary")
    plt.axis('equal')

    for i, n in enumerate(harmonic_list):
        coeffs = calculate_efd(contour, harmonics=n)
        # We don't use normalized coeffs for reconstruction overlay 
        # so they sit on top of the original shape
        rx, ry = reconstruct(coeffs)

        plt.subplot(1, len(harmonic_list) + 1, i + 2)
        plt.plot(contour[:, 0], contour[:, 1], 'k--', alpha=0.2)
        plt.plot(rx, ry, 'r')
        plt.title(f"Harmonics: {n}")
        plt.axis('equal')

    plt.tight_layout()
    plt.show()

# # --- Example Usage ---
# # Creating a dummy "star" shape with noise to simulate a complicated boundary
# t_orig = np.linspace(0, 2*np.pi, 100)
# r = 10 + 3*np.sin(5*t_orig) + np.random.normal(0, 0.2, 100)
# x_orig = r * np.cos(t_orig)
# y_orig = r * np.sin(t_orig)
# sample_contour = np.column_stack([x_orig, y_orig])

# visualize_harmonics(sample_contour)


