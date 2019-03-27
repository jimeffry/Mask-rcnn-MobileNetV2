import json
import os
import sys
from collections import defaultdict
import itertools
import numpy as np
ROOT_DIR = os.path.abspath("../../")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

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

    def add_image(self, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_coco(self, dataset_dir, subset, year="2014",class_names=None):
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
            # All images
            #image_ids = list(coco.imgs.keys())
            pass
        # assign globle values
        self.cat_ids = class_ids
        self.img_ids = image_ids
        self.train_cls_ids = range(len(class_ids)+1)
        # Add classes names
        for i,tmp_id in enumerate(class_ids):
            key_name = coco.loadCats(tmp_id)[0]["name"]
            self.class_id2name[tmp_id] = key_name
            self.name2class_id[key_name] = tmp_id
            self.catId2trainId[tmp_id] = i+1
            self.trainId2catId[i+1] = tmp_id
        # Add image path and annotations
        for i in image_ids:
            self.add_image(
                image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None))
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

def test():
    #dataset = json.load(open('/home/lxy/Downloads/DataSet/COCO-2017/annotations/instances_train2017.json', 'r'))
    dataset = json.load(open('/home/lxy/Downloads/DataSet/COCO-2017/test_annotations/image_info_test2017.json', 'r'))
    print('data keys',dataset.keys())
    cats = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']
    print('annots',len(anns))
    print('imgs',len(imgs))
    print('ann0',anns[0])
    print('img0',imgs[0])
    print('cat0',cats[0])
    cat_curts = []
    all_names = []
    for cat in cats:
        all_names.append(cat['name'])
        if cat['name'] in ['person']:
            cat_curts.append(cat)
    print('all_name:',all_names)
    cat_ids = []
    for cat_ in cat_curts:
        cat_ids.append(cat_['id'])
    print('cat ids ',cat_ids)
    #for ann in anns:
     #   print(ann)
    id=[]
    max_h = max_w=0
    min_h=min_w=10000
    for img in imgs:
        id.append(img['id'])
        if img['height'] > max_h:
            max_h = img['height']
        if img['height'] < min_h:
            min_h = img['height']
        if img['width'] > max_w:
            max_w = img['width']
        if img['width'] < min_w:
            min_w = img['width']
    print("max_h,minh,maxw,minw ",max_h,min_h,max_w,min_w)
    print('img num',len(id))
    sorted(id)
    print('first 5 imgs id:',id[0:5])
    catToImgs = defaultdict(list)
    imgToAnns = defaultdict(list)
    ann_ids = []
    if 'annotations' in dataset and 'categories' in dataset:
        for ann in dataset['annotations']:
            catToImgs[ann['category_id']].append(ann['image_id'])
            imgToAnns[ann['image_id']].append(ann)
            if ann['category_id'] in cat_ids:
                ann_ids.append(ann['id'])
    img_ids = []
    for i, catId in enumerate(cat_ids):
        if i == 0 and len(img_ids) == 0:
            img_ids = set(catToImgs[catId])
        print(i)
    print('img ids: ',len(img_ids))
    ann_ids_ = []
    temp_ids = []
    temp_cat = []
    for imgId in img_ids:
        for ann in imgToAnns[imgId]:
            ann_ids_.append(ann['id'])
            temp_cat.append(ann['category_id'])
            if ann['category_id'] in cat_ids:
                temp_ids.append(ann['id'])
    print(len(ann_ids),len(set(ann_ids_)))
    print(len(temp_ids))
    temp = []
    print(set(temp_cat))

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
    print('w',widths.shape)
    print('x',shifts_x.shape)
    print(shifts_x)
    #print(shifts_y)
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    print('box',box_widths.shape)
    print('center',box_centers_x.shape)
    #print(box_centers_x)
    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2)
    print(box_centers.shape)
    #print(box_centers)
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes

if __name__ == '__main__':
    #generate_anchors(16,[0.5,1,2],[6,6],2,1)
    test()
