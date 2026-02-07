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

import os

def check_os_working():
    print("OS module is working!")
    print(f"Current directory: {os.getcwd()}")
    return True

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


