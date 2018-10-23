# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2018/10/15 16:09
#project: Face recognize
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  papers:
####################################################
import numpy as np 
import math 
import sys
sys.path.append("../")
from mrcnn.mask_rcnn_config import Config as config

def compute_backbone_shapes(image_shape):
    """Computes the width and height of each stage of the backbone network.
    
    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101",'mobilenet']
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    if config.debug_anchor:
        w = box_sizes[:,1]
        h = box_sizes[:,0]
        area = w*h
        print(area[:20])
        idx = np.argmax(area)
        ide = np.argmin(area)
        print('the min',np.min(area))
        print("feature",shape)
        #print("org w,h",widths,heights)
        print("box size",box_sizes[:20])
        print(idx,ide)
        print("max",w[idx],h[idx],np.sqrt(area[idx]))
        print("min",w[ide],h[ide],area[idx])
    return boxes


def generate_pyramid_anchors(image_shape):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    feature_shapes = compute_backbone_shapes(image_shape)
    scales = config.RPN_ANCHOR_SCALES
    ratios = config.RPN_ANCHOR_RATIOS
    feature_strides = config.BACKBONE_STRIDES
    anchor_stride =config.RPN_ANCHOR_STRIDE
    assert len(scales)==len(feature_shapes),"every pyramid has a salce,so scales are equal to srides"
    for i in range(len(scales)):
        print(scales[i])
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)

if __name__=='__main__':
    image_shape = [640,640]
    a= generate_pyramid_anchors(image_shape)
    print(a.shape)