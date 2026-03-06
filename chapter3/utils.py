#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:57:24 2024

@author: qilin

store some functions in train3.py

"""
import torch
import numpy as np
import os
from thop import profile
import time
from pycocotools import mask as maskUtils
import cv2
from pycocotools.cocoeval import COCOeval
import json

def print_losses(loss_dict, lr):
    print(f"Total Loss: {sum(loss_dict.values()):.4f}")
    print(f"  BBox Loss: {loss_dict['loss_box_reg']:.4f}")
    print(f"  Segmentation Loss: {loss_dict['loss_mask']:.4f}")
    print(f"  Classification Loss: {loss_dict['loss_classifier']:.4f}")
    print(f"  RPN Box Loss: {loss_dict['loss_objectness']:.4f}")
    print(f"Learning Rate: {lr:.6f}")

def get_flops(model, inputs):
    flops, params = profile(model, inputs=inputs)
    return flops, params
    
def measure_fps(model, data_loader, device):
    model.eval()
    num_images = 0
    total_time = 0.0
    with torch.no_grad():
        for images, _ in data_loader:
            images = list(img.to(device) for img in images)
            
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            num_images += len(images)
            
            total_time += end_time - start_time
    fps = num_images / total_time
    print(f"FPS: {fps:.2f}")

def draw_predictions(images, outputs, targets, epoch, output_dir, class_names, filename):
    for image, output in zip(images, outputs):
        image_id = targets[0]['image_id'].item()
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)

        # Convert to RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        # Draw bounding boxes and masks
        for i, (box, score, label, mask) in enumerate(zip(output['boxes'], output['scores'], output['labels'], output['masks'])):
            if score < 0.7:  # Thresholding score for visualization
                continue
            box = box.tolist()
            score = score.item()
            label = label.item()
            mask = mask[0].mul(255).byte().cpu().numpy()
            # Apply threshold to mask
            mask = mask > 128
            mask_encoded = maskUtils.encode(np.asfortranarray(mask))
            mask_encoded['counts'] = mask_encoded['counts'].decode('utf-8')
            
            
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0, 255), 2)
            
            # Add class name and score to the image
            class_name = class_names[0]
            text = f'{class_name}: {score:.2f}'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = int((box[0] + box[2]) / 2 - text_size[0] / 2)
            text_y = int((box[1] + box[3]) / 2 + text_size[1] / 2)
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)

            # Create a transparent mask overlay
            mask_overlay = np.zeros_like(image, dtype=np.uint8)
            mask_overlay[mask] = (0, 255, 0, 128)
            image = cv2.addWeighted(image, 1.0, mask_overlay, 0.5, 0)
            
          
        # Save the image with annotations
        output_image_path = os.path.join(output_dir, f"epoch_{epoch}_{image_id}.png")
        cv2.imwrite(output_image_path, image)



def save_model(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def save_predictions(cocoDt, epoch, path):
    with open(path, 'w') as f:
        json.dump(cocoDt, f)