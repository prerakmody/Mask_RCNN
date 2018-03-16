import os
import cv2
import json
import pprint
import colorsys
import skimage.io
import numpy as np
import skimage.transform
from shapely import geometry
from descartes import PolygonPatch
import matplotlib.pyplot as plt

from src.config import Config
import src.utils as utils

class MapillaryConfig(Config):
    NAME = "mapillary"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    NUM_CLASSES = 1 + 14  # background + 3 shapes
    # [ '1' '17' '23' '24' '25' '26' '27' '28' '29' '30' '31' '32' '33' '8']
    
    IMAGE_MAX_DIM = 512
    
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    
    TRAIN_ROIS_PER_IMAGE = 30
    ROI_POSITIVE_RATIO = 0.9
    
    STEPS_PER_EPOCH = 2250
    VALIDATION_STEPS = 2


class MapillaryDataset(utils.Dataset):
    
    def __init__(self, url_dataset, mapper, config, data_type, image_ids = []):
        
        # Base Code (copied) 
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        
        # Add config
        self.config = config
        
        # Read Mapper
        with open(mapper, 'r') as fp:
            self.mapper_json = json.load(fp)
        
        # Add classes
        self.add_classes()
        
        
        self.url_train = ''
        self.url_train_images = ''
        
        # Add Raw Images (.jpg)
        if data_type == 'train': 
            self.url_train = os.path.join(url_dataset, 'mapillary-vistas-dataset_public_v1.0/training')
            self.url_train_images = os.path.join(self.url_train, 'images')
        elif data_type == 'val':
            self.url_train = os.path.join(url_dataset, 'mapillary-vistas-dataset_public_v1.0/validation')
            self.url_train_images = os.path.join(self.url_train, 'images')
        elif data_type == 'test':
            self.url_train = os.path.join(url_dataset, 'mapillary-vistas-dataset_public_v1.0_test/testing')
            self.url_train_images = os.path.join(self.url_train, 'images')
        
        iter_test = -1
        if len(image_ids): # for testing purposes
            iter_test = 0
        
        if self.url_train_images != '':
            images_list = os.listdir(self.url_train_images)
            print ('Mode : {0} has {1} images'.format(data_type, len(images_list)))
            for i, image in enumerate(images_list):
                url_train_tmp = os.path.join(self.url_train_images, image)
                if len(image_ids): # for testing purposes
                    if i in image_ids:
                        self.add_image(source=self.dataset,
                                   image_id=iter_test,
                                   path=url_train_tmp)
                        iter_test += 1
                else:
                    self.add_image(source=self.dataset,
                                   image_id=i,
                                   path=url_train_tmp)
        
            # Prepare
            self.prepare()
        else:
            print ('Problemo!')
        
        
    
    def add_classes(self):
        self.dataset = 'mapillary'
        
        self.add_class(self.dataset, 1, "vehicle-car")
        self.add_class(self.dataset, 2, "vehicle-caravan")
        self.add_class(self.dataset, 3, "vehicle-truck")
        self.add_class(self.dataset, 4, "vehicle-bus")
        self.add_class(self.dataset, 5, "vehicle-trailer")
        self.add_class(self.dataset, 6, "vehicle-motorcycle")
        self.add_class(self.dataset, 7, "vehicle-bicycle")
        self.add_class(self.dataset, 8, "vehicle-train")
        self.add_class(self.dataset, 9, "flat-sidewalk")
        self.add_class(self.dataset, 19, "human-person")
        self.add_class(self.dataset, 20, "human-rider")
        self.add_class(self.dataset, 21, "sky")
        self.add_class(self.dataset, 24, "object-pole")
        self.add_class(self.dataset, 28, "void-ego-vehicle")
         
    def load_image(self, image_id, show=False):
        image = skimage.io.imread(self.image_info[image_id]['path'])
        if show:
            f, axarr = plt.subplots(1, figsize=(10,10))
            frame1 = plt.gca()
            axarr.imshow(image)
        return self.helper_resize_image_to_MAX_DIM(image)
    
    def load_mask(self, image_id, show=False, verbose=False, test=False):
        masks_binary = self.helper_create_mask(image_id, verbose, test)
        
        masks = []
        class_ids_orig = []
        class_ids = []
        
        inst_id_global = 0
        if len(masks_binary): 
            for class_id in masks_binary:
                for instance_id in masks_binary[class_id]:
                    mask_class_instance = np.array(masks_binary[class_id][instance_id])
                    mask_class_instance = self.helper_resize_image_to_MAX_DIM(mask_class_instance)
                    if len(np.unique(mask_class_instance)) > 1:
                        masks.append(mask_class_instance)
                        class_ids.append(self.map_source_class_id(self.dataset + '.' + class_id))
                    inst_id_global += 1

            masks = np.array(masks).transpose(1, 2, 0)
            class_ids = np.array(class_ids).astype(np.uint32)
        
        if show:
            self.helper_check_masks(image_id, masks_binary, verbose, test)
        
        return masks, class_ids
    
    def helper_create_mask(self, image_id, verbose, test):
        label_name  = self.image_info[image_id]['path'].split('/')[-1].split('.jpg')[0] + '.png'
        label_url   = os.path.join(self.url_train, 'instances', label_name)
        if verbose:
            print (' - Instance URL : {0}'.format(label_url))
        if os.path.exists(label_url):
            img_int8  = cv2.imread(label_url)
            img_int16 = skimage.io.imread(label_url)
            class_ids = np.unique(img_int8)
            class_instance_ids = np.unique(img_int16)
            h, w = img_int16.shape
            
            # 1.1 Find the percentage of pixels for each instance_id
            class_instance_ids_count = np.unique(img_int16, return_counts=True)
            tot_pixels = float(w * h)
            class_instance_ids_count = {i:round((j/tot_pixels)*100.0,2) for i,j in zip(class_instance_ids_count[0], class_instance_ids_count[1])}

            # 1.2 Loop over all class_instance_ids. Create mapping of int8:[int16]
            class_ids_mapping_16_8 = {class_id : [] for class_id in class_ids}
            for class_instance_id in class_instance_ids:
                # 1.1.1 Remove instances that are small
                if class_instance_ids_count[class_instance_id] <= 0.01: # mask-size check
                    img_int16[img_int16 == class_instance_id] = 0
                else:
                    for class_id in class_ids_mapping_16_8:
                        playment_class = self.mapper_json['mapillary_class'][str(class_id)]['playment_class']
                        class_floor, class_cieling = class_id*256, (class_id + 1)*256
                        if class_instance_id >= class_id*256 and class_instance_id < (class_id + 1)*256:
                            # 1.3 remove (class, instance) pair that we dont consider training for
                            if int(playment_class) == 0:
                                img_int16[img_int16 == class_instance_id] = 0
                            else:
                                class_ids_mapping_16_8[class_id].append(class_instance_id)   
            # pprint.pprint(class_ids_mapping_16_8)

            # 1.4 Remove those class_ids not having any instances
            class_ids_mapping_16_8_copy = class_ids_mapping_16_8.copy()
            for class_id in class_ids_mapping_16_8_copy:
                if class_ids_mapping_16_8_copy[class_id] == []:
                    class_ids_mapping_16_8.pop(class_id)
            
            if test: 
                # pprint.pprint(class_instance_ids_count)
                pass
                
            # 2. Using class_ids_mapping_16_8, create multiple binary masks
            masks_binary_res = {}
            masks_count = 0
            for class_id_int8 in sorted(class_ids_mapping_16_8):
                if len(class_ids_mapping_16_8[class_id_int8]) == 0:
                    class_ids_mapping_16_8[class_id_int8] = [class_id_int8 * 256]

                for inst_id, class_id_int16 in enumerate(sorted(class_ids_mapping_16_8[class_id_int8])):
                    tmp = np.zeros((img_int16.shape), dtype=np.uint8)
                    tmp[img_int16 != int(class_id_int16)] = 0
                    tmp[img_int16 == int(class_id_int16)] = 1
                    unique_elems = np.unique(tmp)
                    if len(unique_elems) > 1: # this might be a redundant check
                        if test:
                            pass
                        class_id_playment = self.mapper_json['mapillary_class'][str(class_id_int8)]['playment_class']
                        masks_count += 1
                        if str(class_id_playment) not in masks_binary_res:
                            masks_binary_res[class_id_playment] = {}   
                        tmp_name = '{0}'.format(len(masks_binary_res[class_id_playment]))
                        masks_binary_res[class_id_playment][tmp_name] = tmp
            
            if verbose:
                print (' - Total Masks : ', masks_count)
                print (' - Playment Classes : ', {int(class_id) : len(masks_binary_res[class_id]) for class_id in masks_binary_res})
            
            return masks_binary_res
        else:
            return []
    
    def helper_check_masks(self, image_id, masks_binary, verbose, test):
        f, axarr = plt.subplots(1, figsize=(15,15))
        frame1 = plt.gca()
        
        label_name  = self.image_info[image_id]['path'].split('/')[-1].split('.jpg')[0] + '.png'
        label_url   = os.path.join(self.url_train, 'instances', label_name)
        axarr.imshow(skimage.io.imread(label_url), cmap='gray', vmin=0, vmax = 65535)
        
        masks_binary_printed_count = 0
        masks_binary_count = len([1 for class_id in masks_binary for instance_id in masks_binary[class_id]])
        
        # Colors
        N = len(masks_binary.keys())
        brightness = 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        
        for i, class_id_city in enumerate(sorted(masks_binary)):
            for instance_id in masks_binary[class_id_city]:
                tmp = np.array(masks_binary[class_id_city][instance_id])
                try:
                    _, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Step1 : hierarchy[0][x] = [next, prev, child, parent]
                    hierarchy_idx = {}
                    for idx, contour_info in enumerate(hierarchy[0]):
                        if contour_info[3] == -1: # if no parent
                            hierarchy_idx[idx] = {}
                        else: # if parent
                            hierarchy_idx[contour_info[3]] = {}
                            
                    # Step2 : loop over the different non-parent polygons
                    for idx in hierarchy_idx:
                        contour_pts = [list(pt[0]) for pt in contours[idx]]
                        contour_pts = np.array(contour_pts)
                    
                        if len(contour_pts) >= 3:
                            # if int(class_id_city) in [8]:
                            masks_binary_printed_count += 1
                            poly = geometry.Polygon(contour_pts)
                            patch = PolygonPatch(poly, facecolor = colors[i], edgecolor = colors[i], alpha = 0.9)
                            # patch = PolygonPatch(poly, facecolor = [0,1,0], edgecolor = [0,0,0], alpha = 0.9)
                            axarr.add_patch(patch)
                        else:
                            if test:
                                print (' --> Unable to draw : Class ID : {0} || Instance ID : {1}'.format(class_id_city, instance_id))
                                print (contour_pts)
                    
                    # Step3 : Draw BBs
                    horizontal_indicies = np.where(np.any(tmp, axis=0))[0]
                    vertical_indicies   = np.where(np.any(tmp, axis=1))[0]
                    if horizontal_indicies.shape[0]:
                        x1, x2 = horizontal_indicies[[0, -1]]
                        y1, y2 = vertical_indicies[[0, -1]]
                        x2 += 1
                        y2 += 1
                    else:
                        x1, x2, y1, y2 = 0, 0, 0, 0
                    
                    if int(class_id_city) != 21: #sky
                        poly = geometry.box(x1, y1, x2, y2) #(minx, miny, maxx, maxy)
                        patch = PolygonPatch(poly, facecolor = colors[i], edgecolor = colors[i], alpha = 0.3)
                        axarr.add_patch(patch)
                    caption = '{0}_{1}'.format(class_id_city, instance_id)
                    axarr.text(x1, y1 + 8, caption, color='w', size=11, backgroundcolor="none")
                    
                    

                except Exception as e:
                    print (' --> Error:', e)
        
        if verbose:
            print (' - Total Printed Masks : ', masks_binary_printed_count)
    
    def helper_resize_image_to_MAX_DIM(self, image, show=False, verbose=False):
        if verbose:
            print ('-->', img.shape, list(np.unique(img)))
        
        # Image dimensions
        if len(image.shape) == 3:
            (r, c, ch) = image.shape
        elif len(image.shape) == 2:
            (r, c)   = image.shape
        
        # Resize image to have max dimension MAX_DIM
        if r > c:
            new_r = self.config.IMAGE_MAX_DIM
            new_c = round(c / r * self.config.IMAGE_MAX_DIM)
        else:
            new_c = self.config.IMAGE_MAX_DIM
            new_r = round(r / c * self.config.IMAGE_MAX_DIM)
        resized_image = skimage.transform.resize(image, (new_r, new_c), preserve_range=True, mode='reflect').astype('uint8')
        
        if verbose:
            print ('-------->', img_trans.shape, list(np.unique(img_trans)))
        if show:
            f,axarr = plt.subplots(1,2, figsize=(15,15))
            axarr[0].imshow(img)
            axarr[1].imshow(img_trans)
        
        return resized_image

if __name__ == "__main__":
    pass
#     url_dataset = '/home/play/GOD_DATASET/open_datasets/mapillary'
#     mapillary_mapper = '/home/play/playment/production/Mask_RCNN/demo/raw/merge__cityscapes_mapillary_v1.json'
#     mapillary_config = MapillaryConfig()
#     trainData = MapillaryDataset(url_dataset, mapillary_train_images_list_txt_file, mapillary_mapper, mapillary_config)