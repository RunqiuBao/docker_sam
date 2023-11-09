import numpy
import torch
from transformers import SamModel, SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(device)

from datasets import load_dataset

dataset = load_dataset("nielsr/breast-cancer", split="train")

import cv2
import numpy as np
from PIL import Image

idx = 10

# load image
image = dataset[idx]["image"]

cv2.imwrite("/root/data/sample.png", numpy.array(image))

# load ground truth segmentation
ground_truth_seg = np.array(dataset[idx]["label"])
np.unique(ground_truth_seg)

# a random color palette mapping each class to a color (RGB)
palette = [[120, 120, 120], [6, 230, 230]]

color_seg = np.zeros((ground_truth_seg.shape[0], ground_truth_seg.shape[1], 3), dtype=np.uint8) # height, width, 3

for label, color in enumerate(palette):
    color_seg[ground_truth_seg == label, :] = color

cv2.imwrite("/root/data/mask.png", color_seg)

def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

input_boxes = get_bounding_box(ground_truth_seg)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare image + box prompt for the model
inputs = processor(image, input_boxes=[[[input_boxes]]], return_tensors="pt").to(device)
for k,v in inputs.items():
  print(k,v.shape)

model.to(device)

# forward pass
# note that the authors use `multimask_output=False` when performing inference
with torch.no_grad():
  outputs = model(**inputs, multimask_output=False)

# apply sigmoid
medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

cv2.imwrite("/root/data/inference.png", medsam_seg * 255)
