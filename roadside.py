#!/usr/bin/env python
# coding: utf-8

"""
Docstring for roadside.py

Aubrey Moore (aubreymoore2013@gmail.com)
Last modified: 2026-02-07

This module provides python functions for building automated detection of coconut rhinoceros beetle 
damage in digital images.
"""

from ultralytics.models.sam import SAM3SemanticPredictor
# import cv2
import numpy as np
import torch
import os
import exif
import pandas as pd
import sqlite3
from pathlib import Path
import gc
from time import sleep
import shapely
from shapely.geometry import Polygon



# import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.spatial.distance import euclidean

import os
# import json
import pandas as pd
from pathlib import Path
import sqlite3
# from icecream import ic
# import glob




# # Usage example:
# check_os_working()

def conv_poly_from_array_to_wkt(poly: np.array) -> str:
    return Polygon(poly).wkt

# # Usage example:

# poly_wkt = 'POLYGON((0 0, 0 40, 40 40, 40 0, 0 0))'
# poly = conv_poly_from_wkt_to_array(poly_wkt)
# ic(conv_poly_from_array_to_wkt(poly));


def conv_poly_from_wkt_to_array(poly_wkt: str) -> np.array:
    return np.array(shapely.from_wkt(poly_wkt).exterior.coords)

# # Usage example:

# poly_wkt = 'POLYGON((0 0, 0 40, 40 40, 40 0, 0 0))'
# ic(conv_poly_from_wkt_to_array(poly_wkt));


# def pretty_size(size):
# 	"""Pretty prints a torch.Size object"""
# 	assert(isinstance(size, torch.Size))
# 	return " × ".join(map(str, size))

# def dump_tensors(gpu_only=True):
# 	"""Prints a list of the Tensors being tracked by the garbage collector."""
# 	import gc
# 	total_size = 0
# 	for obj in gc.get_objects():
# 		try:
# 			if torch.is_tensor(obj):
# 				if not gpu_only or obj.is_cuda:
# 					print("%s:%s%s %s" % (type(obj).__name__, 
# 										  " GPU" if obj.is_cuda else "",
# 										  " pinned" if obj.is_pinned else "",
# 										  pretty_size(obj.size())))
# 					total_size += obj.numel()
# 			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
# 				if not gpu_only or obj.is_cuda:
# 					print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
# 												   type(obj.data).__name__, 
# 												   " GPU" if obj.is_cuda else "",
# 												   " pinned" if obj.data.is_pinned else "",
# 												   " grad" if obj.requires_grad else "", 
# 												   " volatile" if obj.volatile else "",
# 												   pretty_size(obj.data.size())))
# 					total_size += obj.data.numel()
# 		except Exception as e:
# 			pass        
# 	print("Total size:", total_size)

# Usage example:

# dump_tensors()



def check_gpu():
    """ 
    Checks for GPU availability and prints CUDA version and GPU device name if available.
    Returns True if GPU is available, otherwise False.
    """
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("No GPU available.")
        return False



# def list_gpu_tensors():
#     for obj in gc.get_objects():
#         try:
#             # Check if object is a tensor and on a GPU
#             if torch.is_tensor(obj) and obj.is_cuda:
#                 print(f"Type: {type(obj).__name__:20} | Shape: {tuple(obj.shape)!s:20} | Device: {obj.device}")
#             # Also check for tensors hidden inside objects (like model parameters)
#             elif hasattr(obj, 'data') and torch.is_tensor(obj.data) and obj.data.is_cuda:
#                  print(f"Type: {type(obj).__name__:20} | Shape: {tuple(obj.data.shape)!s:20} | Device: {obj.data.device}")
#         except Exception:
#             pass



# def delete_results_from_gpu_memory():
#     """
#     Explicitly manages memory after processing each image to prevent running out of GPU memory
#     """
#     global results_gpu
#     del results_gpu 
#     gc.collect() 
#     torch.cuda.empty_cache() # Clears unoccupied cached memory

# Usage example:

# delete_results_from_gpu_memory()



def run_sam3_semantic_predictor(input_image_path, text_prompts):
    # Initialize predictor with configuration
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model="sam3.pt",
        half=True,  # Use FP16 for faster inference
        save=True,  # Save image visualizing output results
        save_txt=False,  # Save output results in text format
        save_conf=False,  # Save confidence scores   
        imgsz=1932,  # Adjusted image size from 1920 to meet stride 14 requirement
        batch=1,
        device="0",  # Use GPU device 0
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # Set image once for multiple queries
    predictor.set_image(input_image_path)

    # Query with multiple text prompts
    results = predictor(text=text_prompts)

    # Visualize and save objects
    # visualize_objects(results)
    return results

# # Example usage:

# root_dir = "/home/aubrey/Desktop/sam3-2026-01-31"
# image_paths = ["20251129_152106.jpg", "08hs-palms-03-zglw-superJumbo.webp"]
# text_prompts = ["coconut palm tree"]

# os.chdir(root_dir) # ensure we start in the correct directory
# for image_path in image_paths:
#     results_gpu = run_sam3_semantic_predictor(image_path, text_prompts)

#     # Free up GPU memory in preparation for detecting objects in the next image
#     # This is a work-around to prevent out-of-memory errors from the GPU
#     # I move all results for further processing and use the GPU only for object detection.
#     print('deleting results from GPU memory')       
#     results_cpu = [r.cpu() for r in results_gpu] # copy results to CPU
#     delete_results_from_gpu_memory()

# print("Processing complete.")



# These functions are used for building a SQLite database of images and their detected objects

def update_images_table(image_path, db_path):
    """ 
    Saves data for a single image into an 'images' table in a SQLite database. 
    If the image contains embedded EXIF metadata, GIS coordinates are extracted and saved. and saves it in a SQLite database.
    If the database exists, one record is appended. Otherwise, a new database is created before adding the record.

    Args:
        image_path (str): Path to the image file.
        db_path (str): Path to the SQLite database file.
    Returns:
        None
    """

    with open(image_path, 'rb') as f:
        imgx = exif.Image(f)

    if imgx.has_exif:

        # to see all available exif_data  use imgx.get_all()

        exif_data = {
            'image_path': image_path,
            'image_width': imgx.image_width,
            'image_height': imgx.image_height,
            'timestamp': imgx.datetime
        }

        d, m, s = imgx.gps_latitude
        latitude = d + m/60 + s/3600   
        if imgx.gps_latitude_ref == 'S':
            latitude = -latitude  
        exif_data['latitude']  = latitude   

        d, m, s = imgx.gps_longitude
        longitude = d + m/60 + s/3600   
        if imgx.gps_longitude_ref == 'W':
            longitude = -longitude
        exif_data['longitude'] = longitude
    else:
        exif_data = {
            'image_path': image_path,
            'image_width': None,
            'image_height': None,
            'timestamp': None,
            'latitude': None,
            'longitude': None
        }
    df_image = pd.DataFrame([exif_data])

    # Connect to the SQLite database (creates if it doesn't exist)
    conn = sqlite3.connect(db_path)
    # cursor = conn.cursor()
    df_image.to_sql(name='images', con=conn, if_exists='append', index=False)
    conn.close()


def update_detections_table(results, image_path, db_path):

    # Process detection results (assuming one image for simplicity: results[0])
    result = results[0]

    # --- Extract Bounding Box Data into a DataFrame ---
    # The .boxes.data attribute is a tensor containing [x_min, y_min, x_max, y_max, confidence, class]
    boxes_data = result.boxes.data.tolist()
    df_boxes = pd.DataFrame(boxes_data, columns=['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id'])

    # Add class names for readability
    # class_names = model.names
    # df_boxes['class_name'] = df_boxes['class_id'].apply(lambda x: class_names[int(x)])

    # --- Extract Segmentation Mask Data ---
    # Masks are more complex as they represent pixel-wise information or polygon points.
    # To put this into a DataFrame, you could store the polygon points list for each object.
    masks_data = []
    # Iterate over each detected object's mask
    for i, mask in enumerate(result.masks.xy):
        # mask.xy contains the polygon points as a list of [x, y] coordinates
        # You can associate this with the corresponding entry in the bounding box DataFrame

        poly_arr = mask
        poly_wkt = conv_poly_from_array_to_wkt(poly_arr)

        masks_data.append({
            'image_path': image_path,
            'object_index': i, 
            'class_id': df_boxes.iloc[i]['class_id'], 
            'poly_wkt': poly_wkt})
        df_masks = pd.DataFrame(masks_data)  

    # merge df_masks and df_detections  
    df_detections = pd.merge(df_masks, df_boxes, how="outer", left_index=True, right_index=True)

    # Connect to the SQLite database (creates if it doesn't exist)
    conn = sqlite3.connect(db_path)
    # cursor = conn.cursor()
    df_detections.to_sql(name='detections', con=conn, if_exists='append', index=False)
    conn.close()


def build_database(image_path, results, db_path):
    """ 
    Builds a SQLite database with 'images' and 'detections' tables from a list of image paths.

    Args:
        image_path (str): Path to an image file.
        results (list): List of detection results from model.
        db_path (str): Path to the SQLite3 database file.
    Returns:
        None
    """
    update_images_table(image_path, db_path)
    update_detections_table(results, image_path, db_path)

##############
# from efd2.py
##############


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



