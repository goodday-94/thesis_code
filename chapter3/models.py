#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:59:02 2024

@author: qilin

store model
"""
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch
from network_files import MaskRCNN, FasterRCNN, AnchorsGenerator
from backbone import resnet50_fpn_backbone, BackboneWithFPN, LastLevelMaxPool
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch.nn as nn
from modelparts import resnet50, resnet50_fuse, resnet50_fuse2, resnet50_fuse3
from torchviz import make_dot
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.detection.backbone_utils import LastLevelMaxPool


######################
# Default Mask-RCNN
######################

def default_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

######################
# Mask-RCNN show each layer
######################

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

def mobilenet_backbone(pretrained):
    # Load a pre-trained MobileNetV2 model
    backbone = torchvision.models.mobilenet_v2(pretrained=pretrained).features
    # MobileNetV2 requires a different number of output channels
    backbone.out_channels = 1280
    return backbone

class EfficientNetBackbone(nn.Module):
    def __init__(self, pretrained):
        super(EfficientNetBackbone, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
        self.out_channels = 1280
    
    def forward(self, x):
        return self.efficientnet.extract_features(x)
    

def default_model2(num_classes):

    backbone = resnet_fpn_backbone('resnet50', pretrained=True)

    # Create the Mask R-CNN model with the custom backbone with FPN
    model = MaskRCNN(backbone, num_classes=num_classes)
    
    # Replace the box predictor with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

######################
# Mask-RCNN code from WZ github
######################

def create_model(num_classes, load_pretrain_weights=False):

    backbone = resnet50_fpn_backbone(pretrain_path="", trainable_layers=5)

    model = MaskRCNN(backbone, num_classes=num_classes)

    if load_pretrain_weights:
        # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_dict = torch.load("./maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
        for k in list(weights_dict.keys()):
            if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))

    return model

def create_model2(num_classes):
    # --- mobilenet_v3_large fpn backbone --- #
    backbone = torchvision.models.mobilenet_v3_large(pretrained=False)
    # print(backbone)
    return_layers = {"features.6": "0",   # stride 8
                     "features.12": "1",  # stride 16
                     "features.16": "2"}  # stride 32
    in_channels_list = [40, 112, 960]
    new_backbone = create_feature_extractor(backbone, return_layers)
    # img = torch.randn(1, 3, 224, 224)
    # outputs = new_backbone(img)
    # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]

    # --- efficientnet_b0 fpn backbone --- #
    #backbone = torchvision.models.efficientnet_b0(pretrained=False)
    #print(backbone)
    #return_layers = {"features.3": "0",  # stride 8
    #                  "features.4": "1",  # stride 16
    #                  "features.8": "2"}  # stride 32
    #in_channels_list = [40, 80, 1280]
    #new_backbone = create_feature_extractor(backbone, return_layers)
    # # img = torch.randn(1, 3, 224, 224)
    # # outputs = new_backbone(img)
    # # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]

    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)
    """
    anchor_sizes = ((64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    model = FasterRCNN(backbone=backbone_with_fpn,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    """
    model = MaskRCNN(backbone_with_fpn, num_classes=num_classes)
    return model


def create_model3(num_classes):
 
    # --- resnet 50 fpn backbone --- #
    #backbone = torchvision.models.resnet50(pretrained=False)
    backbone = resnet50()
    #print(backbone)
    return_layers = {"layer1.2": "0",  # stride 8
                      "layer2.3": "1",  # stride 16
                      "layer3.5": "2",
                      "layer4.2": "3"}  # stride 32
    new_backbone = create_feature_extractor(backbone, return_layers)
    # img = torch.randn(1, 3, 224, 224)
    #outputs = new_backbone(img)
    #[print(f"{k} shape: {v.shape}") for k, v in outputs.items()]
    
    in_channels_list = [256, 512, 1024, 2048]
    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)
    
    model = MaskRCNN(backbone_with_fpn, num_classes=num_classes)
    return model

# Fusion Net, two encoders and do the element-wise adding at stage1-4 in resnet 50
def create_model4(num_classes):
    
    # --- resnet 50 fpn backbone --- #
    #backbone = torchvision.models.resnet50(pretrained=False)
    backbone = resnet50_fuse()
    #print(backbone)
    return_layers = {"encoder1.layer1.2": "0",  # stride 8
                      "encoder1.layer2.3": "1",  # stride 16
                      "encoder1.layer3.5": "2",
                      "encoder1.layer4.2": "3"}  # stride 32
    new_backbone = create_feature_extractor(backbone, return_layers)
    #img = torch.randn(1, 6, 224, 224)
    #outputs = new_backbone(img)
    #[print(f"{k} shape: {v.shape}") for k, v in outputs.items()]
    
    in_channels_list = [256, 512, 1024, 2048]
    #in_channels_list = [256*2, 512*2, 1024*2, 2048*2]
    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)
    
    model = MaskRCNN(backbone_with_fpn, num_classes=num_classes)
    return model


def create_model4_1(num_classes):
    
    # --- resnet 50 fpn backbone --- #
    #backbone = torchvision.models.resnet50(pretrained=False)
    backbone = resnet50_fuse()
    #print(backbone)
    return_layers = {"encoder2.layer1.2": "0",  # stride 8
                      "encoder2.layer2.3": "1",  # stride 16
                      "encoder2.layer3.5": "2",
                      "encoder2.layer4.2": "3"}  # stride 32
    new_backbone = create_feature_extractor(backbone, return_layers)
    #img = torch.randn(1, 6, 224, 224)
    #outputs = new_backbone(img)
    #[print(f"{k} shape: {v.shape}") for k, v in outputs.items()]
    
    in_channels_list = [256, 512, 1024, 2048]
    #in_channels_list = [256*2, 512*2, 1024*2, 2048*2]
    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)
    
    model = MaskRCNN(backbone_with_fpn, num_classes=num_classes)
    return model

# Fusion Net with SE channel-wise attention, two encoders and do the element-wise adding at stage1-4 in resnet 50
def create_model5(num_classes):
    
    # --- resnet 50 fpn backbone --- #
    #backbone = torchvision.models.resnet50(pretrained=False)
    backbone = resnet50_fuse2()
    #print(backbone)
    return_layers = {"add1": "0",  # stride 8
                      "add2": "1",  # stride 16
                      "add3": "2",
                      "add4": "3"}  # stride 32
    new_backbone = create_feature_extractor(backbone, return_layers)
    #img = torch.randn(1, 6, 224, 224)
    #outputs = new_backbone(img)
    #[print(f"{k} shape: {v.shape}") for k, v in outputs.items()]
    
    in_channels_list = [256, 512, 1024, 2048]
    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)
    
    model = MaskRCNN(backbone_with_fpn, num_classes=num_classes)
    return model

def create_model6(num_classes):
    
    # --- resnet 50 fpn backbone --- #
    #backbone = torchvision.models.resnet50(pretrained=False)
    backbone = resnet50_fuse3()
    #print(backbone)
    return_layers = {"cssa1": "0",  # stride 8
                      "cssa2": "1",  # stride 16
                      "cssa3": "2",
                      "cssa4": "3"}  # stride 32
    new_backbone = create_feature_extractor(backbone, return_layers)
    #img = torch.randn(1, 6, 224, 224)
    #outputs = new_backbone(img)
    #[print(f"{k} shape: {v.shape}") for k, v in outputs.items()]
    
    in_channels_list = [256, 512, 1024, 2048]
    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)
    
    model = MaskRCNN(backbone_with_fpn, num_classes=num_classes)
    return model

###############3
###try new code from pytorch
############3##

class Resnet50WithFPN(torch.nn.Module):
    def __init__(self):
        super(Resnet50WithFPN, self).__init__()
        # Get a resnet50 backbone
        # m = resnet50_fuse()
        m = resnet50_fuse()
        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        self.body = create_feature_extractor(m, return_nodes={
                'concat1.cat': '0',
                'concat2.cat': '1',
                'concat3.cat': '2',
                'concat4.cat': '3'})
        # Dry run to get number of channels for FPN
        inp = torch.randn(1, 6, 224, 224)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        #in_channels_list = [256, 512, 1024, 2048]
        # Build FPN
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool())

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x

def create_model7(num_classes):
    model = MaskRCNN(Resnet50WithFPN(), num_classes=num_classes)
    return model


