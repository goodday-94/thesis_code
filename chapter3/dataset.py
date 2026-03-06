#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:58:32 2024

@author: qilin

store dataloader
"""
import torch
from PIL import Image
from pycocotools.coco import COCO
import os
import numpy as np



# Custom dataset class
class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = [ann['bbox'] for ann in anns]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64)
        masks = [self.coco.annToMask(ann) for ann in anns]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        target = {}
        nor_boxes = []
        for box in boxes:
            x_min, y_min, width, height = box
            x_max =  width + x_min
            y_max = height + y_min
            nor_boxes.append([x_min, y_min, x_max, y_max])
        target['boxes'] = torch.as_tensor(nor_boxes, dtype=torch.float32)
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = torch.tensor([img_id])

        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, target

    def __len__(self):
        return len(self.ids)


# Custom dataset class
class ConcateDataset(torch.utils.data.Dataset):

    def __init__(self, root1, root2, annotation_file, transforms=None):
        self.root1 = root1
        self.root2 = root2
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        image1 = Image.open(os.path.join(self.root1, path)).convert('RGB')
        image2 = Image.open(os.path.join(self.root2, path)).convert('RGB')
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = [ann['bbox'] for ann in anns]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64)
        masks = [self.coco.annToMask(ann) for ann in anns]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        target = {}
        nor_boxes = []
        for box in boxes:
            x_min, y_min, width, height = box
            x_max =  width + x_min
            y_max = height + y_min
            nor_boxes.append([x_min, y_min, x_max, y_max])
        target['boxes'] = torch.as_tensor(nor_boxes, dtype=torch.float32)
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = torch.tensor([img_id])
        
        if self.transforms is not None:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
            
        image = torch.cat((image1, image2), dim=0)
        return image, target

    def __len__(self):
        return len(self.ids)
    
# Custom dataset class
class ParaDataset(torch.utils.data.Dataset):

    def __init__(self, root1, root2, annotation_file, transforms=None):
        self.root1 = root1
        self.root2 = root2
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        image1 = Image.open(os.path.join(self.root1, path)).convert('RGB')
        image2 = Image.open(os.path.join(self.root2, path)).convert('RGB')
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = [ann['bbox'] for ann in anns]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64)
        masks = [self.coco.annToMask(ann) for ann in anns]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        target = {}
        nor_boxes = []
        for box in boxes:
            x_min, y_min, width, height = box
            x_max =  width + x_min
            y_max = height + y_min
            nor_boxes.append([x_min, y_min, x_max, y_max])
        target['boxes'] = torch.as_tensor(nor_boxes, dtype=torch.float32)
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = torch.tensor([img_id])
        
        if self.transforms is not None:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
            
        return (image1, image2), target

    def __len__(self):
        return len(self.ids)