import os
import sys
import random
import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
import time
import colorsys
from skimage.measure import find_contours
from matplotlib import patches
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
from maskconfig import config as maskconfig
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
from skimage import img_as_ubyte



class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

class MaskInference(object):
    def __init__(self,model_path,log_path):
        self.model_path = model_path
        model_dir = log_path
        config = InferenceConfig()
        self.config = config
        self.model_net = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
        self.load_weight()
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

    def load_weight(self):
        self.model_net.load_weights(self.model_path,by_name=True)

    def detectfrompath(self,img_path):
        self.images = [cv2.imread(img_path)]
        t1 = time.time()
        self.detect_results = self.model_net.detect(self.images,verbose=1)
        if maskconfig.time :
            print("one mask image consumes time: ",time.time()-t1)

    def detectfromimg(self,img_list):
        self.images = img_list
        t1 = time.time()
        self.detect_results = self.model_net.detect(self.images,verbose=1)
        if maskconfig.time :
            print("one mask image consumes time: ",time.time()-t1)

    def apply_mask(self,image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                    image[:, :, c] *(1 - alpha) + alpha * color[c] * 255,
                                    image[:, :, c])
        return image

    def random_colors(self,N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors
        
    def show_mask(self,title="show",figsize=(10, 10),
                      show_mask=True, show_bbox=True,colors=None, captions=None):
        """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        title: (optional) Figure title
        show_mask, show_bbox: To show masks and bounding boxes or not
        figsize: (optional) the size of the image
        colors: (optional) An array or colors to use with each object
        captions: (optional) A list of strings to use as captions for each object
        """
        # Number of instances
        fg_num = len(self.detect_results)
        figure_nums = range(fg_num)
        fig_cur = plt.figure(num=0,figsize=figsize)
        cols = 4
        gs = gridspec.GridSpec(fg_num // cols + 1,cols)
        gs.update(hspace=0.4)
        ax = []
        for idx in figure_nums:
            row = (idx // cols)
            col = idx % cols
            boxes = self.detect_results[idx]['rois']
            masks = self.detect_results[idx]['masks']
            class_ids = self.detect_results[idx]['class_ids']
            scores = self.detect_results[idx]['scores']
            N = boxes.shape[0]
            if not N:
                print("\n*** No instances to display *** \n")
            else:
                assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
            # If no axis is passed, create one and automatically call show()
            ax.append(fig_cur.add_subplot(gs[row,col]))
            # Generate random colors
            colors = colors or self.random_colors(N)
            # Show area outside image boundaries.
            height, width = self.images[idx].shape[:2]
            ax[-1].set_ylim(height + 10, -10)
            ax[-1].set_xlim(-10, width + 10)
            ax[-1].axis('off')
            #ax[-1].set_title(title)
            masked_image = self.images[idx].astype(np.uint32).copy()
            for i in range(N):
                color = colors[i]
                # Bounding box
                if not np.any(boxes[i]):
                    # Skip this instance. Has no bbox. Likely lost in image cropping.
                    continue
                y1, x1, y2, x2 = boxes[i]
                if show_bbox:
                    p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                        alpha=0.7, linestyle="dashed",
                                        edgecolor=color, facecolor='none')
                    ax[-1].add_patch(p)
                # Label
                if not captions:
                    class_id = class_ids[i]
                    score = scores[i] if scores is not None else None
                    label = self.class_names[class_id]
                    x = random.randint(x1, (x1 + x2) // 2)
                    caption = "{} {:.3f}".format(label, score) if score else label
                else:
                    caption = captions[i]
                ax[-1].text(x1, y1 + 8, caption,
                        color='w', size=11, backgroundcolor="none")
                # Mask
                mask = masks[:, :, i]
                if show_mask:
                    masked_image = self.apply_mask(masked_image, mask, color)
                # Mask Polygon
                # Pad to ensure proper polygons for masks that touch image edges.
                padded_mask = np.zeros(
                    (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = mask
                #contours = find_contours(padded_mask, 0.5)
                # the input must be binary image
                _,contours, hierarchy = cv2.findContours(padded_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
                for verts in contours:
                    # Subtract the padding and flip (y, x) to (x, y)
                    #verts = np.fliplr(verts) - 1
                    verts = np.reshape(verts,[-1,2])
                    p = Polygon(verts, facecolor="none", edgecolor=color)
                    ax[-1].add_patch(p)
            ax[-1].imshow(masked_image.astype(np.uint8))
        plt.show()

    def swapbackground(self,img_list,gd_path,save_paths):
        img_gd_org = skimage.io.imread(gd_path)
        self.detectfromimg(img_list)
        for idx in range(len(self.detect_results)):
            save_path = save_paths[idx]
            image_save = self.images[idx]
            fg_h, fg_w = image_save.shape[:2]
            h,w = img_gd_org.shape[:2]
            if maskconfig.debug:
                print("org img gd ",img_gd_org[0,:10,0])
            if h != fg_h or w != fg_w :
                img_gd = img_as_ubyte(skimage.transform.resize(img_gd_org, (fg_h, fg_w),mode='constant'))
                if maskconfig.debug:
                    print("resize gd ",img_gd[0,:10,0])
            else:
                img_gd = img_gd_org
            detect_results = self.detect_results[idx]
            boxes = detect_results['rois']
            masks = detect_results['masks']
            class_ids = detect_results['class_ids']
            scores = detect_results['scores']
            N = boxes.shape[0]
            if not N:
                print("\n*** No instances to display *** \n")
                continue
            else:
                assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
            for i in range(N):
                if not np.any(boxes[i]):
                    continue
                class_idx = class_ids[i]
                label = self.class_names[class_idx]
                score = scores[i] if scores is not None else 0.0
                if label == 'person' and score >= maskconfig.threshold :
                    mask = masks[:,:,i]
                    mask = np.expand_dims(mask,axis=2)
                    image_save = image_save * mask
                    image_gd = img_gd *(1-mask)
                    image_save = image_save+image_gd
                    image_save = image_save.astype(np.uint8)
                    save_img_path = save_path + '_'+str(i)+'.jpg'
                    skimage.io.imsave(save_img_path,image_save)

if __name__ == '__main__':
    #image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    # Local path to trained weights file
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    #image = "/home/lxy/Develop/faster_rcnn/Mask_RCNN/images/8239308689_efa6c11b08_z.jpg"
    image1 = cv2.imread("/home/lxy/Develop/faster_rcnn/Mask_RCNN/images/3975_20.jpg")
    image2 = cv2.imread("/home/lxy/Develop/faster_rcnn/Mask_RCNN/images/8973_0.jpg")
    image_ground = "/home/lxy/Develop/faster_rcnn/Mask_RCNN/images/background.jpg"
    save_p1 = "/home/lxy/Develop/faster_rcnn/Mask_RCNN/3975t"
    save_p2 = "/home/lxy/Develop/faster_rcnn/Mask_RCNN/8973t"
    save_ps = [save_p1,save_p2]
    img_list = [image1,image2]
    mask_net = MaskInference(COCO_MODEL_PATH,MODEL_DIR)
    mask_net.swapbackground(img_list,image_ground,save_ps)
    mask_net.show_mask()
    #gd = skimage.io.imread(image_ground)
    #gd = gd[:112,:112,:]
    #skimage.io.imsave("background.jpg",gd)