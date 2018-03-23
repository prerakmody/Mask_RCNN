## STANDARD PYTHON LIBS
import os
import cv2
import json
import skimage.io
import numpy as np
import tensorflow as tf

## CUSTOM LIBS
import src.model as modellib
import src.mapillary as mapillary
from src.config import Config

# Called once
def load_model(ROOT_DIR, device = 'GPU'):
    # return 'abc', []
    device_scope = '/{}:0'.format(device.lower())
    with tf.device(device_scope):
        MODEL_DIR = os.path.join(ROOT_DIR, 'demo', 'model', 'logs')
        class InferenceConfig(mapillary.MapillaryConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        
        inference_config = InferenceConfig()
        print (' - Batch Size : ', inference_config.BATCH_SIZE)

        model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
        # model_path = model.find_last()[1]
        model_path_base = '/home/play/playment/production/Mask_RCNN/demo/model/logs/mapillary20180315T0317/'
        models = ['mask_rcnn_mapillary_0196.h5','mask_rcnn_mapillary_0158.h5', 'mask_rcnn_mapillary_0083.h5']
        model_path = os.path.join(model_path_base, models[0])
        print (' - Model Path : ', model_path)

        if model_path != None:
            model.load_weights(model_path, by_name=True)
            return model, inference_config
        else:
            sys.exit(1)
            return []

################################################
#               PROCESS IMAGE                  #
################################################
def predict(img_url, model, config):
    print ('\n================================================\n')
    
    url_prefix = 'https://'
    if url_prefix not in img_url:
        img_url = url_prefix + img_url
    
    img    = skimage.io.imread(img_url)
    shape_ = img.shape
    if len(shape_) == 3:
        h, w, d = shape_
        img     = helper_img_resize(img, config.IMAGE_MAX_DIM)
        results = model.detect([img], verbose=0)
        masks   = helper_getmasks(results[0], max(h,w))
        return masks
    else:
        return {}
    
def helper_img_resize(img, max_dim):
    if len(img.shape) > 3:
        if img.shape[2] == 3: # RGB IMAGE
            (r, c, ch) = image.shape
            
            # Resize image to have max dimension MAX_DIM
            if r > c:
                new_r = max_dim
                new_c = round(c / r * max_dim)
            else:
                new_c = max_dim
                new_r = round(r / c * max_dim)
            resized_image = skimage.transform.resize(image, (new_r, new_c), preserve_range=True, mode='reflect').astype('uint8')
            
            return resized_image

def helper_img_resize(img, max_dim):
    # Image dimensions
    shape_ = img.shape
    if len(shape_) == 3:
        (r, c, ch) = shape_
    elif len(shape_) == 2:
        (r, c)   = shape_
    # Resize image to have max dimension MAX_DIM
    if r > c:
        new_r = max_dim
        new_c = round(c / r * max_dim)
    else:
        new_c = max_dim
        new_r = round(r / c * max_dim)
    resized_image = skimage.transform.resize(img, (new_r, new_c), preserve_range=True, mode='reflect').astype('uint8')

    return resized_image

def helper_getcontours(img_mask, res_obj, pt_index):
    res_instance = []
    
    if len(img_mask.shape) == 2:
        if list(np.unique(img_mask)) == [0,1]:
            _, contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
            # Step1 : hierarchy[0][x] = [next, prev, child, parent]
            hierarchy_idx = {}
            for idx, contour_info in enumerate(hierarchy[0]):
                if contour_info[3] == -1: # if no parent
                    hierarchy_idx[idx] = {}
                else: # if parent
                    hierarchy_idx[contour_info[3]] = {}

            # Step2 : loop over the different non-parent polygons
            for idx in hierarchy_idx:
                tmp_exterior = []
                contour_pts = [list(pt[0]) for pt in contours[idx]]
                if len(contour_pts) >= 3:
                    for pt in contours[idx]:
                        pt_index_tmp = 'p' + str(pt_index)
                        res_obj['points'][pt_index_tmp] = {'x' : str(pt[0][0]), 'y': str(pt[0][1])}
                        tmp_exterior.append(pt_index_tmp)
                        pt_index += 1
                res_instance.append({'exterior' : tmp_exterior})
    
    return res_instance, res_obj, pt_index

def helper_getmasks(res_predict, max_dim):
    #1. bloat the mask
    #2. cv2.findCountours
    total_masks = res_predict['masks'].shape[2]
    res_obj     = {'image_height' : -1, 'image_width' : -1, 'image_url' : '', 'instances' : [], 'points' : {}}
    
    pt_index = 0
    for mask_id in range(total_masks):
        print ('Mask making progress : ', mask_id, '/', total_masks, end='\r')
        tmp_mask              = res_predict['masks'][:,:,mask_id]
        tmp_mask              = helper_img_resize(tmp_mask, max_dim)
        tmp_class_id          = res_predict['class_ids'][mask_id]
        # tmp_class_name        = class_names[tmp_class_id]
        res_instance, res_obj, pt_index = helper_getcontours(tmp_mask, res_obj, pt_index)
        tmp_instance          = {'label' : str(tmp_class_id), 'segments' : res_instance}
        res_obj['instances'].append(tmp_instance)
    
    return res_obj