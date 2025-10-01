import os
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt
import math

def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def crop_to_petri_dish(image):

    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours:
        return image, (0, 0, image.shape[1], image.shape[0])  # Return original if no contours
    
    # Get bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])
    side_length = max(w, h)
    x_center, y_center = x + w // 2, y + h // 2
    
    x_new = max(0, x_center - side_length // 2)
    y_new = max(0, y_center - side_length // 2)
    
    cropped_image = image[y_new:y_new + side_length, x_new:x_new + side_length]
    return cropped_image, (x_new, y_new, side_length)

def padder(image, patch_size):

    h, w = image.shape[:2]

    height_padding = math.ceil(h / patch_size) * patch_size - h
    width_padding = math.ceil(w / patch_size) * patch_size - w

    top_padding = height_padding // 2
    bottom_padding = height_padding - top_padding
    left_padding = width_padding // 2
    right_padding = width_padding - left_padding

    padded_image = cv2.copyMakeBorder(
        image, 
        top_padding, 
        bottom_padding, 
        left_padding, 
        right_padding, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )
    return padded_image

def divide_mask_into_sections(predicted_mask, sections=5):
    height, width = predicted_mask.shape
    section_width = width // sections
    divided_sections = []
    
    for i in range(sections):
        start_x = i * section_width
        end_x = (i + 1) * section_width if i != sections - 1 else width  # Ensure the last section captures remaining pixels
        divided_sections.append(predicted_mask[:, start_x:end_x])
    
    return divided_sections

def load_mask(filepath):
    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0  # Convert to binary
    return mask

def extract_rsa(mask):

    # Skeletonize the mask to simplify root structure
    skeleton = skeletonize(mask)

    # Label connected components in the skeleton
    labeled_skeleton = label(skeleton)

    # Extract properties of labeled regions
    properties = regionprops(labeled_skeleton)

    root_data = []

    # Calculate geometric properties for all roots
    for prop in properties:
        root_length = prop.perimeter
        root_width = prop.major_axis_length
        root_data.append({
            "length": root_length,
            "width": root_width,
            "centroid": prop.centroid,
            "type": "secondary"  # Default to secondary root
        })

    # Determine the primary root based on the longest length
    if root_data:
        primary_root = max(root_data, key=lambda x: x["length"])
        primary_root["type"] = "primary"  # Mark the primary root

    return root_data, skeleton






