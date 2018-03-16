
# coding: utf-8

# In[1]:


## STANDARD PYTHON LIBS
import os
import sys
import numpy as np
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

## ADDING TO ROOT
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(ROOT_DIR)

## CUSTOM LIBS
import src.utils as utils
import src.model as modellib
from src.config import Config
import src.visualize as visualize

# GPU LIBS
import keras
import tensorflow as tf

cuda_version = os.popen("cat /usr/local/cuda/version.txt ").read()
print ('TF : ', tf.__version__, '  Keras : ', keras.__version__, '  CUDA : ', cuda_version)


# In[2]:


if utils.check_gpu(verbose=0):
    pass
else:
    pass
    # sys.exit(1)


# In[3]:


if 'src.mapillary' in sys.modules : del sys.modules['src.mapillary']
if 'src.utils'     in sys.modules : del sys.modules['src.utils']
if 'src.model'     in sys.modules : del sys.modules['src.model']

import src.utils as utils
import src.model as modellib
import src.mapillary as mapillary

# if __name__ == "__main__":
url_dataset = '/home/play/datasets/open_datasets/mapillary'
mapillary_mapper = '/home/play/playment/Mask_RCNN/demo/raw/merge__cityscapes_mapillary_v2.json'
mapillary_config = mapillary.MapillaryConfig()
mapillary_config.IMAGES_PER_GPU = 8
mapillary_config.STEPS_PER_EPOCH = 10
# trainData = mapillary.MapillaryDataset(url_dataset, mapillary_mapper, mapillary_config, data_type = 'train')
# valData   = mapillary.MapillaryDataset(url_dataset, mapillary_mapper, mapillary_config, data_type = 'val')


# In[4]:


# idx, show, verbose, test = 2459, True, True, True
# idx, show, verbose, test = 12488, True, True, True
# idx, show, verbose, test = 0, False, False, False
# img = trainData.load_image(idx, show=show)
# masks, class_ids = trainData.load_mask(idx, show=show, verbose=verbose, test=test)


# In[5]:


# train_generator = modellib.data_generator(trainData, mapillary_config, shuffle=True, batch_size=mapillary_config.BATCH_SIZE)
# val_generator   = modellib.data_generator(valData, mapillary_config, shuffle=True, batch_size=mapillary_config.BATCH_SIZE,augment=False)
# input_, output_ =  next(train_generator)


# ## TRAINING

# In[6]:


if 'src.model' in sys.modules : del sys.modules['src.model']
if 'src.utils'     in sys.modules : del sys.modules['src.utils']
import src.utils as utils
import src.model as modellib

    
MODEL_DIR = os.path.join(ROOT_DIR, 'demo', 'model', 'logs')
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'demo', 'model', "mask_rcnn_coco.h5")
model = modellib.MaskRCNN(mode="training", config=mapillary_config, model_dir=MODEL_DIR)

# init_with = "coco"  # imagenet, coco, or last
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    model_path = model.find_last()[1]
    print (' - Path : ', model_path)
    model.load_weights(model_path, by_name=True)


# In[ ]:
trainData = '/home/play/datasets/open_datasets/mapillary_hdf5/mapillary-vistas-dataset_public_v1.0/training'
valData   = '/home/play/datasets/open_datasets/mapillary_hdf5/mapillary-vistas-dataset_public_v1.0/validation'

model.train(trainData, valData, 
            learning_rate=mapillary_config.LEARNING_RATE, 
            epochs=8, 
            layers='heads')

print ('-----------------------------------------------------------')
# # In[ ]:


# # sys.exit(1)


# # # INFERENCE

# # In[ ]:


# if 'src.model' in sys.modules : del sys.modules['src.model']
# import src.model as modellib
# class InferenceConfig(mapillary.MapillaryConfig):
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1

# inference_config = InferenceConfig()
# print (' - Batch Size : ', inference_config.BATCH_SIZE)
# model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
# model_path = model.find_last()[1]
# print (' - Model Path : ', model_path)

# if model_path != None:
#     model.load_weights(model_path, by_name=True)
# else:
#     sys.exit(1)


# # In[ ]:


# if 'src.mapillary' in sys.modules : del sys.modules['src.mapillary']
# import src.mapillary as mapillary
# testData = mapillary.MapillaryDataset(url_dataset, mapillary_mapper, mapillary_config, data_type = 'test')


# # In[ ]:


# def get_ax(rows=1, cols=1, size=8):
#     _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
#     return ax

# test_img = testData.load_image(45)
# results  = model.detect([test_img], verbose=1)
# r = results[0]

# if 'src.visualize' in sys.modules : del sys.modules['src.visualize']
# import src.visualize as visualize
# visualize.display_instances(test_img, r['rois'], r['masks'], r['class_ids'], 
#                             valData.class_names, r['scores'], ax=get_ax())


# # # SCRATCHPAD

