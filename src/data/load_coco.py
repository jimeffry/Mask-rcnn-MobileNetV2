# -*- coding: utf-8 -*-
###############################################
#created by :  lxy
#Time:  2019/1/22 10:09
#project: mask rcnn
#rversion: 0.1
#tool:   python 2.7
#modified:
#description  papers:
####################################################
import json
import os
import sys
from collections import defaultdict
import itertools
import numpy as np
import cv2
sys.path.append('/home/lxy/software/cocoapi/PythonAPI/')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

class CocoDataset(object):
    def __init__(self):
        self.class_id2name = {0:"BG"}
        self.name2class_id = {"BG":0}
        self.image_info = []
        self.cat_ids = []
        self.img_ids = []
        self.train_cls_ids = []
        self.catId2trainId = {}
        self.trainId2catId = {}
        self.trainId2Names = {}
        self.CocoDatasetNames = []
        self.ValidNames = []

    def add_image(self, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_coco(self, dataset_dir, subset, year="2014",class_names=None,areaRng=[]):
        """
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_names: If provided, only loads images that have the given classes.
        CoCoapi:
        loadCats(cat_id) -> [categories_dict]
        loadAnns(ann_id) -> [annotation_dict]
        loadImgs(img_id) -> [imgs_dict]
        getCatIds() -> [categories_ids]
        getAnnIds() -> [annottation_ids]
        getImgIds() -> [imgs_ids]
        # on hole data
        coco.imgs -> img_dict
        coco.cats -> cats_dict
        coco.anns -> ann_dict
        """
        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)
        #load dataset names
        for tmp_key in coco.cats.keys():
            tmp_name = coco.cats[tmp_key]['name']
            self.CocoDatasetNames.append(tmp_name)
        # Load all classes or a subset
        if class_names is not None:
            if type(class_names) == list:
                flt_names = class_names
            else:
                flt_names = [class_names]
            class_ids = sorted(coco.getCatIds(catNms=flt_names))
        else:
            class_ids = sorted(coco.getCatIds())
        # All images or a subset
        if len(class_ids)>0:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            #image_ids = list(coco.imgs.keys())
            pass
        valid_img_ids = []
        if len(areaRng) >0:
            val_class_ids = []
            tmp_ann_list = []
            for tmp_id in image_ids:
                tmp_ann_ids = coco.getAnnIds(imgIds=[tmp_id], catIds=class_ids,areaRng=areaRng, iscrowd=None)
                if len(tmp_ann_ids) >0:
                    valid_img_ids.append(tmp_id)
                    tmp_anns = coco.loadAnns(tmp_ann_ids)
                    tmp_ann_list.extend(tmp_anns)
                else:
                    continue
            for tmp_ann in tmp_ann_list:
                val_class_ids.extend([tmp_ann['category_id']])
            val_class_ids = list(set(val_class_ids))
        else:
            valid_img_ids = image_ids
            val_class_ids = class_ids
        # assign globle values
        self.cat_ids = val_class_ids
        self.img_ids = valid_img_ids
        self.train_cls_ids = range(len(val_class_ids)+1)
        #get valid names for input names
        self.ValidNames = [coco.cats[c_id]['name'] for c_id in self.cat_ids ]
        # Add classes names
        for i,tmp_id in enumerate(self.cat_ids):
            key_name = coco.loadCats(tmp_id)[0]["name"]
            self.class_id2name[tmp_id] = key_name
            self.name2class_id[key_name] = tmp_id
            self.catId2trainId[tmp_id] = i+1
            self.trainId2catId[i+1] = tmp_id
            self.trainId2Names[i+1] = key_name
        # Add image path and annotations
        for idx in self.img_ids:
            self.add_image(
                image_id=idx,
                path=os.path.join(image_dir, coco.imgs[idx]['file_name']),
                width=coco.imgs[idx]["width"],
                height=coco.imgs[idx]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[idx], catIds=self.cat_ids,areaRng=areaRng,iscrowd=None))
                )

    def load_mask(self, img_list_id):
        """Load instance masks for the given image.
        the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of train class IDs of the instance masks.
        """
        img_info_dict = self.image_info[img_list_id]
        instance_masks = []
        train_cls_ids = []
        annotations = img_info_dict["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for ann in annotations:
            tmp_cls_id = self.catId2trainId[ann['category_id']]
            if tmp_cls_id:
                m = self.annToMask(ann, img_info_dict["height"],img_info_dict["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if ann['iscrowd']:
                    # Use negative class ID for crowds
                    tmp_cls_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != img_info_dict["height"] or m.shape[1] != img_info_dict["width"]:
                        m = np.ones([img_info_dict["height"], img_info_dict["width"]], dtype=bool)
                instance_masks.append(m)
                train_cls_ids.append(tmp_cls_id)

        # Pack instance masks into an array
        if len(train_cls_ids)>0:
            masks = np.stack(instance_masks, axis=2).astype(np.bool)
            train_cls_ids = np.array(train_cls_ids, dtype=np.int32)
            return masks, train_cls_ids
        else:
            # Call super class to return an empty mask
            return None,None

    def load_bbox(self,img_list_id):
        img_info_dict = self.image_info[img_list_id]
        bboxes = []
        train_cls_ids = []
        annotations =  img_info_dict["annotations"]
        #print('load_bbox id',img_list_id,img_info_dict['id'])
        for ann in annotations:
            tmp_cls_id = self.catId2trainId[ann['category_id']]
            bb = ann['bbox']
            x1, x2, y1, y2 = [np.ceil(bb[0]), np.ceil(bb[0]+bb[2]), np.ceil(bb[1]), np.ceil(bb[1]+bb[3])]
            bboxes.append([x1,y1,x2,y2])
            train_cls_ids.append(tmp_cls_id)
        if len(bboxes)>0:
            bboxes = np.stack(bboxes).astype(np.int32)
            train_cls_ids = np.array(train_cls_ids,dtype=np.int32)
            return bboxes,train_cls_ids
        else:
            None,None

    def load_image(self, img_list_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        img_path = self.image_info[img_list_id]['path']
        #print('load_path',img_list_id,img_path)
        #print('load_img id',self.image_info[img_list_id]['id'])
        image = cv2.imread(img_path)
        # If grayscale. Convert to RGB for consistency.
        if image is None:
            return None
        if len(image.shape) != 3:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m