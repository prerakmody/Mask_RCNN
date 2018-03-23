
# coding: utf-8

# In[1]:


## STANDARD PYTHON LIBS
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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


# if utils.check_gpu(verbose=0):
#     pass
# else:
#     sys.exit(1)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True 
# k.tensorflow_backend.set_session(tf.Session(config=config))


# In[17]:


if 'src.mapillary' in sys.modules : del sys.modules['src.mapillary']
if 'src.utils'     in sys.modules : del sys.modules['src.utils']
if 'src.model'     in sys.modules : del sys.modules['src.model']

import src.utils as utils
import src.model as modellib
import src.mapillary as mapillary

if __name__ == "__main__":
## OPTION1 : CACHED
     mapillary_config = mapillary.MapillaryConfig(images_per_gpu=4, gpu_count = 4)
     mapillary_config.STEPS_PER_EPOCH  = 1#1125
     mapillary_config.VALIDATION_STEPS = 1#50
     print ('BATCH SIZE : ', mapillary_config.BATCH_SIZE)
     trainData = '/home/ubuntu/datasets/open_datasets/mapillary_hdf5_16/mapillary-vistas-dataset_public_v1.0/training'
     valData   = '/home/ubuntu/datasets/open_datasets/mapillary_hdf5_16/mapillary-vistas-dataset_public_v1.0/validation'
   
    
#    mapillary_config = mapillary.MapillaryConfig(images_per_gpu=16, gpu_count = 4)
#    mapillary_config.GPU_IMPL_TYPE = 'self'
#    mapillary_config.STEPS_PER_EPOCH  = 1#280
#    mapillary_config.VALIDATION_STEPS = 1#12
#    print ('BATCH SIZE : ', mapillary_config.BATCH_SIZE)
#    trainData = '/home/ubuntu/datasets/open_datasets/mapillary_hdf5_64/mapillary-vistas-dataset_public_v1.0/training'
#    valData   = '/home/ubuntu/datasets/open_datasets/mapillary_hdf5_64/mapillary-vistas-dataset_public_v1.0/validation'

#     mapillary_config = mapillary.MapillaryConfig(images_per_gpu=16, gpu_count = 1)
#     mapillary_config.STEPS_PER_EPOCH  = 1
#     mapillary_config.VALIDATION_STEPS = 1
#     print ('BATCH SIZE : ', mapillary_config.BATCH_SIZE)
#     trainData = '/home/ubuntu/datasets/open_datasets/mapillary_hdf5_64/mapillary-vistas-dataset_public_v1.0/training'
#     valData   = '/home/ubuntu/datasets/open_datasets/mapillary_hdf5_64/mapillary-vistas-dataset_public_v1.0/validation'

    
#     mapillary_config = mapillary.MapillaryConfig(images_per_gpu=16, gpu_count = 8)
#     mapillary_config.STEPS_PER_EPOCH  = 140
#     mapillary_config.VALIDATION_STEPS = 6
#     print ('BATCH SIZE : ', mapillary_config.BATCH_SIZE)
    
    
    ## OPTION2 : GENERATE ON-THE--FLY
#     url_dataset      = '/home/ubuntu/datasets/open_datasets/mapillary'
#     mapillary_mapper = '/home/ubuntu/playment/Mask_RCNN/demo/raw/merge__cityscapes_mapillary_v2.json'
#     trainData = mapillary.MapillaryDataset(url_dataset, mapillary_mapper, mapillary_config, data_type = 'train')
#     valData   = mapillary.MapillaryDataset(url_dataset, mapillary_mapper, mapillary_config, data_type = 'val')


# In[ ]:


# idx, show, verbose, test = 2459, True, True, True
# idx, show, verbose, test = 12488, True, True, True
# idx, show, verbose, test = 0, False, False, False
# img = trainData.load_image(idx, show=show)
# masks, class_ids = trainData.load_mask(idx, show=show, verbose=verbose, test=test)


# ## TRAINING

# In[18]:


if 'src.model'            in sys.modules : del sys.modules['src.model']
if 'src.utils'            in sys.modules : del sys.modules['src.utils']
if 'src.parallel_model'   in sys.modules : del sys.modules['src.parallel_model']
import src.utils as utils
import src.model as modellib
import src.parallel_model as parallel_model

if 'keras.' in sys.modules : del sys.modules['keras']
import keras

    
MODEL_DIR         = os.path.join(ROOT_DIR, 'demo', 'model', 'logs')
COCO_MODEL_PATH   = os.path.join(ROOT_DIR, 'demo', 'model', "mask_rcnn_coco.h5")
model             = modellib.MaskRCNN(mode="training", config=mapillary_config, model_dir=MODEL_DIR)
# model.keras_model = parallel_model.ParallelModel(model.keras_model, mapillary_config.GPU_COUNT, verbose = 0)
# model.keras_model.summary()

init_with = "coco"  # imagenet, coco, or last
# init_with = "last"  # imagenet, coco, or last

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


## OPTION1
if os.path.exists(trainData) & os.path.exists(valData): 
    model.train(trainData, valData, 
                learning_rate=mapillary_config.LEARNING_RATE, 
                epochs=1, 
                layers='heads')
else:
    print ('os.path.exists(trainData) : ', os.path.exists(trainData))
    print ('os.path.exists(valData)   : ', os.path.exists(valData))

## OPTION2 
# model.train(trainData, valData, 
#             learning_rate=mapillary_config.LEARNING_RATE, 
#             epochs=158, 
#             layers='heads')


# In[ ]:


sys.exit(1)


# # INFERENCE

# In[ ]:


if 'src.model' in sys.modules : del sys.modules['src.model']
import src.model as modellib

MODEL_DIR = os.path.join(ROOT_DIR, 'demo', 'model', 'logs')

class InferenceConfig(mapillary.MapillaryConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
print (' - Batch Size : ', inference_config.BATCH_SIZE)
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
model_path = model.find_last()[1]
print (' - Model Path : ', model_path)

if model_path != None:
    model.load_weights(model_path, by_name=True)
else:
    sys.exit(1)


# In[ ]:


if 'src.mapillary' in sys.modules : del sys.modules['src.mapillary']
import src.mapillary as mapillary
testData = mapillary.MapillaryDataset(url_dataset, mapillary_mapper, mapillary_config, data_type = 'test')


# In[ ]:


def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(15,15))
    return ax

test_img = testData.load_image(1586)
results  = model.detect([test_img], verbose=1)
r = results[0]

if 'src.visualize' in sys.modules : del sys.modules['src.visualize']
import src.visualize as visualize
visualize.display_instances(test_img, r['rois'], r['masks'], r['class_ids'], 
                            testData.class_names, r['scores'], ax=get_ax())


# # SCRATCHPAD

# In[ ]:


print (model.keras_model)
gen = modellib.data_generator_play(trainData)
ip, op = next(gen)


# In[ ]:


model.keras_model(ip)


# In[ ]:





