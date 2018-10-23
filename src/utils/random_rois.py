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


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network
    would generate.

    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer ground truth class IDs
    gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

    Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(0.9 * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        per_idx = 0
        box_rois = []
        while True:
            size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            # delta here is the offset of box center
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)
            #show this way: nx1 = max(x1+w/2-size/2+delta_x)
            nx1 = int(max(gt_x1 + w / 2 + delta_x - size / 2, 0))
            #show this way: ny1 = max(y1+h/2-size/2+delta_y)
            ny1 = int(max(gt_y1 + h / 2 + delta_y - size / 2, 0))
            nx2 = nx1 + size
            ny2 = ny1 + size
            if nx2 > width-1 or ny2 > height-1:
                continue 
            per_idx+=1
            box_rois.append([ny1,nx1,ny2,nx2])
            if per_idx == rois_per_box:
                break
        box_rois = np.vstack(box_rois)
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois
    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    box_rois = []
    for i in range(remaining_count):
        size = np.random.randint(config.min_size, min(w, h) / 2)
        #top_left
        x1 = np.random.randint(0, w - size)
        y1 = np.random.randint(0, h - size)
        x2 = x1+size
        y2 = y1+size
        box_rois.append([y1,x1,y2,x2])
    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    global_rois = np.vstack(box_rois)
    rois[-remaining_count:] = global_rois
    return rois