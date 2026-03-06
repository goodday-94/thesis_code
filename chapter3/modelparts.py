#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 22:34:32 2024

@author: qilin

save model parts\
    
"""

import torch
import torch.nn as nn
from torchviz import make_dot
from torchsummary import summary
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

# Custom module for concatenation
class Concat(nn.Module):
    def __init__(self, dim):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)

# Custom module for element-wise addition
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, input1, input2):
        return (input1 + input2)/2

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # change 3 to 6 to get the concate images
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            ) # using stride 2 to implement pooling

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



##################
# Early fusion, Fusenet 
##################

class ResNetDetailed(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNetDetailed, self).__init__()
        self.in_channels = 64

        # Initial convolution and batch normalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # _Initial convolution and batch normalization
        self.conv1_ = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_ = nn.BatchNorm2d(64)
        self.relu_ = nn.ReLU(inplace=True)
        self.maxpool_ = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1
        self.conv2_1 = block(64, 64, stride=1)
        self.conv2_2 = block(64 * block.expansion, 64, stride=1)
        self.conv2_3 = block(64 * block.expansion, 64, stride=1)
        
        # _Layer 1
        self.conv2_1_ = block(64, 64, stride=1)
        self.conv2_2_ = block(64 * block.expansion, 64, stride=1)
        self.conv2_3_ = block(64 * block.expansion, 64, stride=1)

        # Layer 2
        self.conv3_1 = block(64 * block.expansion, 128, stride=2, downsample=self._downsample_layer(64 * block.expansion, 128 * block.expansion, stride=2))
        self.conv3_2 = block(128 * block.expansion, 128, stride=1)
        self.conv3_3 = block(128 * block.expansion, 128, stride=1)
        self.conv3_4 = block(128 * block.expansion, 128, stride=1)
        
        # _Layer 2
        self.conv3_1_ = block(64 * block.expansion, 128, stride=2, downsample=self._downsample_layer(64 * block.expansion, 128 * block.expansion, stride=2))
        self.conv3_2_ = block(128 * block.expansion, 128, stride=1)
        self.conv3_3_ = block(128 * block.expansion, 128, stride=1)
        self.conv3_4_ = block(128 * block.expansion, 128, stride=1)

        # Layer 3
        self.conv4_1 = block(128 * block.expansion, 256, stride=2, downsample=self._downsample_layer(128 * block.expansion, 256 * block.expansion, stride=2))
        self.conv4_2 = block(256 * block.expansion, 256, stride=1)
        self.conv4_3 = block(256 * block.expansion, 256, stride=1)
        self.conv4_4 = block(256 * block.expansion, 256, stride=1)
        self.conv4_5 = block(256 * block.expansion, 256, stride=1)
        self.conv4_6 = block(256 * block.expansion, 256, stride=1)
        
        # _Layer 3
        self.conv4_1_ = block(128 * block.expansion, 256, stride=2, downsample=self._downsample_layer(128 * block.expansion, 256 * block.expansion, stride=2))
        self.conv4_2_ = block(256 * block.expansion, 256, stride=1)
        self.conv4_3_ = block(256 * block.expansion, 256, stride=1)
        self.conv4_4_ = block(256 * block.expansion, 256, stride=1)
        self.conv4_5_ = block(256 * block.expansion, 256, stride=1)
        self.conv4_6_ = block(256 * block.expansion, 256, stride=1)

        # Layer 4
        self.conv5_1 = block(256 * block.expansion, 512, stride=2, downsample=self._downsample_layer(256 * block.expansion, 512 * block.expansion, stride=2))
        self.conv5_2 = block(512 * block.expansion, 512, stride=1)
        self.conv5_3 = block(512 * block.expansion, 512, stride=1)
        
        # _Layer 4
        self.conv5_1_ = block(256 * block.expansion, 512, stride=2, downsample=self._downsample_layer(256 * block.expansion, 512 * block.expansion, stride=2))
        self.conv5_2_ = block(512 * block.expansion, 512, stride=1)
        self.conv5_3_ = block(512 * block.expansion, 512, stride=1)

        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # Split the input tensor into two 3-channel images
        x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        print(x1.shape)
        print(x2.shape)
        # Initial layers
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        
        # _Initial layers
        x2 = self.conv1_(x2)
        x2 = self.bn1_(x2)
        x2 = self.relu_(x2)
        x2 = self.maxpool_(x2)

        # Layer 1
        x1 = self.conv2_1(x1)
        x1 = self.conv2_2(x1)
        x1 = self.conv2_3(x1)
        
        # _Layer 1
        x2 = self.conv2_1_(x2)
        x2 = self.conv2_2_(x2)
        x2 = self.conv2_3_(x2)
        x2 = torch.add(x2, x1)

        # Layer 2
        x1 = self.conv3_1(x1)
        x1 = self.conv3_2(x1)
        x1 = self.conv3_3(x1)
        x1 = self.conv3_4(x1)
        
        # _Layer 2
        x2 = self.conv3_1_(x2)
        x2 = self.conv3_2_(x2)
        x2 = self.conv3_3_(x2)
        x2 = self.conv3_4_(x2)
        x2 = torch.add(x2, x1)

        # Layer 3
        x1 = self.conv4_1(x1)
        x1 = self.conv4_2(x1)
        x1 = self.conv4_3(x1)
        x1 = self.conv4_4(x1)
        x1 = self.conv4_5(x1)
        x1 = self.conv4_6(x1)
        
        # _Layer 3
        x2 = self.conv4_1_(x2)
        x2 = self.conv4_2_(x2)
        x2 = self.conv4_3_(x2)
        x2 = self.conv4_4_(x2)
        x2 = self.conv4_5_(x2)
        x2 = self.conv4_6_(x2)
        x2 = torch.add(x2, x1)


        # Layer 4
        x1 = self.conv5_1(x1)
        x1 = self.conv5_2(x1)
        x1 = self.conv5_3(x1)
        
        # _Layer 4
        x2 = self.conv5_1_(x2)
        x2 = self.conv5_2_(x2)
        x2 = self.conv5_3_(x2)
        x2 = torch.add(x2, x1)

        # Average pooling and fully connected layer
        x2 = self.avgpool(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc(x2)

        return x2

class DualResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(DualResNet50, self).__init__()
        self.encoder1 = ResNet(block, layers, num_classes)
        self.encoder2 = ResNet(block, layers, num_classes)
        self.concat1 = Concat(dim=1) 
        self.concat2 = Concat(dim=1) 
        self.concat3 = Concat(dim=1) 
        self.concat4 = Concat(dim=1) 
        
    def forward(self, x):
        # Split input into two tensors with 3 channels each
        x1, x2 = torch.split(x, 3, dim=1)

        # Forward pass through both encoders
        out1 = self.encoder1.conv1(x1)
        out1 = self.encoder1.bn1(out1)
        out1 = self.encoder1.relu(out1)
        out1 = self.encoder1.maxpool(out1)

        out2 = self.encoder2.conv1(x2)
        out2 = self.encoder2.bn1(out2)
        out2 = self.encoder2.relu(out2)
        out2 = self.encoder2.maxpool(out2)

        # Element-wise addition of outputs at each layer
        out1 = self.encoder1.layer1(out1)
        out2 = self.encoder2.layer1(out2)
        con1 = self.concat1((out1, out2))

        out1 = self.encoder1.layer2(out1)
        out2 = self.encoder2.layer2(out2)
        con2 = self.concat2((out1, out2))

        out1 = self.encoder1.layer3(out1)
        out2 = self.encoder2.layer3(out2)
        con3 = self.concat3((out1, out2))

        out1 = self.encoder1.layer4(out1)
        out2 = self.encoder2.layer4(out2)
        con4 = self.concat4((out1, out2))

        out2 = self.encoder2.avgpool(out2)
        out2 = torch.flatten(out2, 1)
        out2 = self.encoder2.fc(out2)

        return con1,con2,con3,con4

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        # Squeeze
        y = self.global_avg_pool(x).view(batch_size, num_channels)
        # Excitation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, num_channels, 1, 1)
        # Recalibration
        return x * y.expand_as(x)
    

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out


class DualResNet50_2(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(DualResNet50_2, self).__init__()
        self.encoder1 = ResNet(block, layers, num_classes)
        self.encoder2 = ResNet(block, layers, num_classes)
        
        self.add1 = Add() 
        self.add2 = Add() 
        self.add3 = Add() 
        self.add4 = Add() 

    def forward(self, x):
        # Split input into two tensors with 3 channels each
        x1, x2 = torch.split(x, 3, dim=1)

        # Forward pass through both encoders
        out1 = self.encoder1.conv1(x1)
        out1 = self.encoder1.bn1(out1)
        out1 = self.encoder1.relu(out1)
        out1 = self.encoder1.maxpool(out1)

        out2 = self.encoder2.conv1(x2)
        out2 = self.encoder2.bn1(out2)
        out2 = self.encoder2.relu(out2)
        out2 = self.encoder2.maxpool(out2)

        # Element-wise addition of outputs at each layer
        out1 = self.encoder1.layer1(out1)
        out2 = self.encoder2.layer1(out2)
        con1 = self.add1(out1, out2)

        out1 = self.encoder1.layer2(out1)
        out2 = self.encoder2.layer2(out2)
        con2 = self.add2(out1, out2)
        
        out1 = self.encoder1.layer3(out1)
        out2 = self.encoder2.layer3(out2)
        con3 = self.add3(out1, out2)

        out1 = self.encoder1.layer4(out1)
        out2 = self.encoder2.layer4(out2)
        con4 = self.add4(out1, out2)
        
        out1 = self.encoder1.avgpool(out1)
        out1 = torch.flatten(out1, 1)
        out1 = self.encoder1.fc(out1)
        
        out2 = self.encoder1.avgpool(out2)
        out2 = torch.flatten(out2, 1)
        out2 = self.encoder2.fc(out2)

        return con1, con2, con3, con4, out1, out2


class ECABlock(torch.nn.Module):
  def __init__(self, kernel_size=3, channel_first=None):
    super().__init__()

    self.channel_first = channel_first

    self.GAP = torch.nn.AdaptiveAvgPool2d(1)
    self.f = torch.nn.Conv1d(1, 1, kernel_size=kernel_size, padding = kernel_size // 2, bias=False)
    self.sigmoid = torch.nn.Sigmoid()


  def forward(self, x):

    x = self.GAP(x)

    # need to squeeze 4d tensor to 3d & transpose so convolution happens correctly
    x = x.squeeze(-1).transpose(-1, -2)
    x = self.f(x)
    x = x.transpose(-1, -2).unsqueeze(-1) # return to correct shape, reverse ops

    x = self.sigmoid(x)

    return x

class ChannelSwitching(torch.nn.Module):
  def __init__(self, switching_thresh):
    super().__init__()
    self.k = switching_thresh

  def forward(self, x, x_prime, w):

    self.mask = w < self.k
     # If self.mask is True, take from x_prime; otherwise, keep x's value
    x = torch.where(self.mask, x_prime, x)

    return x

class SpatialAttention(torch.nn.Module):

  def __init__(self):
    super().__init__()

    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, rgb_feats, ir_feats):
    # get shape
    B, C, H, W = rgb_feats.shape

    # channel concatenation (x_cat -> B,2C,H,W)
    x_cat = torch.cat((rgb_feats, ir_feats), axis=1)

    # create w_avg attention map (w_avg -> B,1,H,W)
    cap = torch.mean(x_cat, dim=1)
    w_avg = self.sigmoid(cap)
    w_avg = w_avg.unsqueeze(1)

    # create w_max attention maps (w_max -> B,1,H,W)
    cmp = torch.max(x_cat, dim=1)[0]
    w_max = self.sigmoid(cmp)
    w_max = w_max.unsqueeze(1)

    # weighted feature map (x_cat_w -> B,2C,H,W)
    x_cat_w = x_cat * w_avg * w_max

    # split weighted feature map (x_ir_w, x_rgb_w -> B,C,H,W)
    x_rgb_w = x_cat_w[:,:C,:,:]
    x_ir_w = x_cat_w[:,C:,:,:]

    # fuse feature maps (x_fused -> B,H,W,C)
    x_fused = (x_ir_w + x_rgb_w)/2

    return x_fused

class CSSA(torch.nn.Module):

  def __init__(self, switching_thresh=0.005, kernel_size=3, channel_first=None):
    super().__init__()

    # self.eca = ECABlock(kernel_size=kernel_size, channel_first=channel_first)
    self.eca_rgb = ECABlock(kernel_size=kernel_size, channel_first=channel_first)
    self.eca_ir = ECABlock(kernel_size=kernel_size, channel_first=channel_first)
    self.cs = ChannelSwitching(switching_thresh=switching_thresh)
    self.sa = SpatialAttention()

  def forward(self, rgb_input, ir_input):
    # channel switching for RGB input
    rgb_w = self.eca_rgb(rgb_input)
    rgb_feats = self.cs(rgb_input, ir_input, rgb_w)

    # channel switching for IR input
    ir_w = self.eca_ir(ir_input)
    ir_feats = self.cs(ir_input, rgb_input, ir_w)

    # spatial attention
    fused_feats = self.sa(rgb_feats, ir_feats)

    return fused_feats

class DualResNet50_3(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(DualResNet50_3, self).__init__()
        self.encoder1 = ResNet(block, layers, num_classes)
        self.encoder2 = ResNet(block, layers, num_classes)
    
        self.cssa1 = CSSA()
        self.cssa2 = CSSA()
        self.cssa3 = CSSA()
        self.cssa4 = CSSA()

    def forward(self, x):
        # Split input into two tensors with 3 channels each
        x1, x2 = torch.split(x, 3, dim=1)

        # Forward pass through both encoders
        out1 = self.encoder1.conv1(x1)
        out1 = self.encoder1.bn1(out1)
        out1 = self.encoder1.relu(out1)
        out1 = self.encoder1.maxpool(out1)

        out2 = self.encoder2.conv1(x2)
        out2 = self.encoder2.bn1(out2)
        out2 = self.encoder2.relu(out2)
        out2 = self.encoder2.maxpool(out2)

        # Element-wise addition of outputs at each layer
        out1 = self.encoder1.layer1(out1)
        out2 = self.encoder2.layer1(out2)
        con1 = self.cssa1(out1, out2)

        out1 = self.encoder1.layer2(out1)
        out2 = self.encoder2.layer2(out2)
        con2 = self.cssa2(out1, out2)
        
        out1 = self.encoder1.layer3(out1)
        out2 = self.encoder2.layer3(out2)
        con3 = self.cssa3(out1, out2)

        out1 = self.encoder1.layer4(out1)
        out2 = self.encoder2.layer4(out2)
        con4 = self.cssa4(out1, out2)
        
        out1 = self.encoder1.avgpool(out1)
        out1 = torch.flatten(out1, 1)
        out1 = self.encoder1.fc(out1)
        
        out2 = self.encoder1.avgpool(out2)
        out2 = torch.flatten(out2, 1)
        out2 = self.encoder2.fc(out2)

        return con1, con2, con3, con4, out1, out2

def resnet50(num_classes=2):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet50_fuse(num_classes=2):
    return DualResNet50(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet50_fuse2(num_classes=2):
    return DualResNet50_2(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet50_fuse3(num_classes=2):
    return DualResNet50_3(Bottleneck, [3, 4, 6, 3], num_classes)




# Uncomment to test
model = resnet50_fuse()

#img = torch.randn(1, 6, 224, 224)  # Example input with 6 channels
#output = model(img)
#print(output)
train_nodes, eva_nodes = get_graph_node_names(model) 
print(train_nodes)
# Visualize the model structure and save it as a jpg file
# make_dot(output, params=dict(model.named_parameters())).render("model_cssa", format="jpg")

# Print each layer's name and details
#print("Model Layers:")
#for name, layer in model.named_modules():
#    print(f"Layer Name: {name}")
#    print(f"Layer Details: {layer}")
#    print("-" * 50)

# Get all graph node names
#train_nodes, eval_nodes = get_graph_node_names(model) 

# Print node names for inspection
#print("Training nodes:")
#for node in train_nodes:
#    print(node)

#print("\nEvaluation nodes:")
#for node in eval_nodes:
#    print(node)

#summary(model, input_size=(1, 3, 224, 224))
#ot = make_dot(output, params=dict(model.named_parameters()))
#dot.render("show", format="png")
#from IPython.display import Image
#Image(filename='show.png')

