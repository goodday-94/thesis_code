#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 01:31:35 2024

@author: qilin

for single image training

"""
import os
import torch
import numpy as np
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util
import torchvision.transforms as T
from tqdm import tqdm
from utils import print_losses, get_flops, measure_fps, draw_predictions, save_model, save_predictions
from dataset import SingleDataset, ConcateDataset
from models import default_model, create_model, create_model2, create_model3, create_model7

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

# Load datasets
train_img = "replace to dataset root"
test_img = "replace to dataset root"
#train_img_r = "replace to dataset root"
#test_img_r = "replace to dataset root"
train_anno = "replace to dataset root"
test_anno = "replace to dataset root"

dataset_train = SingleDataset(root=train_img, annotation_file=train_anno, transforms=get_transform(train=True))
dataset_val = SingleDataset(root=test_img, annotation_file=test_anno, transforms=get_transform(train=False))

#dataset_train = ConcateDataset(root1=train_img, root2=train_img_r, annotation_file=train_anno, transforms=get_transform(train=True))
#dataset_val = ConcateDataset(root1=test_img, root2=test_img_r, annotation_file=test_anno, transforms=get_transform(train=False))

data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=2, shuffle=True, collate_fn=collate_fn)
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Initialize model, optimizer, and learning rate scheduler
model = create_model3(num_classes=2) # Adjust num_classes as needed
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print('#'*10)
print(device)
print('#'*10)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005) # pretrain lr = 0.005
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

num_epochs = 200

#########
# Train #
#########

for epoch in range(num_epochs):
    model.train()
    epoch_losses = {
    'total': 0,
    'box_reg': 0,
    'mask': 0,
    'classifier': 0,
    'objectness': 0
}
    for images, targets in tqdm(data_loader_train):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Skip images without annotations
        if any(len(t['boxes']) == 0 for t in targets):
            continue
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_losses['total'] += losses.item()
        epoch_losses['box_reg'] += loss_dict['loss_box_reg'].item()
        epoch_losses['mask'] += loss_dict['loss_mask'].item()
        epoch_losses['classifier'] += loss_dict['loss_classifier'].item()
        epoch_losses['objectness'] += loss_dict['loss_objectness'].item()

    lr_scheduler.step()
    
    num_batches = len(data_loader_train)
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print_losses({
        'loss_box_reg': epoch_losses['box_reg'] / num_batches,
        'loss_mask': epoch_losses['mask'] / num_batches,
        'loss_classifier': epoch_losses['classifier'] / num_batches,
        'loss_objectness': epoch_losses['objectness'] / num_batches
    }, optimizer.param_groups[0]['lr'])
    
    # Validation
    model.eval()
    cocoGt = dataset_val.coco
    cocoDt = []
    category_names = ["patch"]
    
    ########
    # Test #
    ########
    
    # Every 5 epoche evaluation
    if (epoch+1)%10 == 0: 
        with torch.no_grad():
            for i, (images, targets) in enumerate(tqdm(data_loader_val)):
                images = list(img.to(device) for img in images)
                outputs = model(images)
    
                for target, output in zip(targets, outputs):
                    image_id = target['image_id'].item()
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()
                    masks = output['masks'].cpu().numpy()
                    
                    #image_info = cocoGt.loadImgs(image_id)[0]
                    #path = image_info['file_name']
                    #image = Image.open(os.path.join(dataset_val.root, path)).convert('RGB')
                    # record images
                    filename = f"{epoch + 1}_{i + 1}.png"
                    output_dir = r"/home/qilin/Task_lane/patch detection/selfcode/logs/predictions/"
                    #draw_predictions(images, outputs, targets, epoch, output_dir, category_names, filename)
                    
                    for box, score, label, mask in zip(boxes, scores, labels, masks):
                        box = box.tolist()
                        mask = (mask[0] > 0.5).astype(np.uint8)
                        
                        rle = mask_util.encode(np.asfortranarray(mask))
                        if isinstance(rle['counts'], bytes):
                            rle['counts'] = rle['counts'].decode('utf-8')
                        
                        cocoDt.append({
                            'image_id': image_id,
                            'category_id': int(label),
                            'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]], # Convert to [x, y, width, height]
                            'score': float(score),
                            'segmentation': rle
                        })
        
        # Saving results
        prediction_path = os.path.join(output_dir, f"2984_2985_acd_epoch_{epoch + 1}.json")  
        #save_predictions(cocoDt, epoch, prediction_path)
        
        # Show evaluation
        cocoDt = cocoGt.loadRes(cocoDt)
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    
    # Save model state
    if epoch == num_epochs-1:  # Save every 10 epochs
        save_model(epoch, model, optimizer, f"2984_2985_acd_{epoch + 1}.pth")
    
    # Print model states
    if epoch == 0 or epoch == num_epochs-1:
        sample_inputs = torch.randn(1, 3, 1014, 1248).to(device)
        flops, params = get_flops(model, (sample_inputs,))
        flops = flops/1e9
        params = params/1e6
        print(f"Model Parameters(M): {params}")
        print(f"FLOPs(G): {flops}")
        
        # Measure FPS
        measure_fps(model, data_loader_val, device)