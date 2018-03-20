# SYNC To S3
- aws s3 sync ./mapillary_hdf5_16/ s3://open_datasets/open_datasets/mapillary_hdf5_16/

# SYNC DOWN FROM S3
 - aws s3 sync s3://open_datasets . --exclude '*/COCO/*' --exclude '*/Cityscapes/*' --exclude '*/KITTI/*' --exclude '*/Udacity/*' --exclude '*/configs/*' --exclude '*/proprietary_datasets/*' --exclude '*/mapillary/*'