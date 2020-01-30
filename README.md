# Head and Neck Segmentation Stuff
Retrieve DICOM data from using data retriever, then run the following scripts in order. 

### 0.0 dcm_to_masks.py  
sorts through .mat files in ./cts/cts and ./contours/contours and returns CT and contour libraries
not efficient but gets job done

### 0.1 npydata/clean.py  
cropping and adjusting window/level of entire dataset

### 0.2 sort.py  
Throw out images not containing a desired structure. Leave some data out for testing.

### 0.3 data_check.py  
visualize training data

### 1.0/1.1 data.py, unet.py  
2d unet and dataset are defined. random B-spline augmentation with bspline kwarg.

### 2.0 train.py  
initialize the model and begin training

### 2.1 update.py  
continue training an existing model

### 3.0 deploy.py / threshold.py  
deploy the model on test set. visualize contours and loss. get optimum threshold.
