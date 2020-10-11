# libraries used
!pip install git+https://github.com/qubvel/segmentation_models
!pip install -U git+https://github.com/albu/albumentations
numpy
keras
tensorflow==1.14.0
sklearn
skimage
cv2
optparse

# Resize the images. Also include only those images in the training and testing set, which has corresponding
# binary output images. There are ~200 images in the training dataset which does not have corresponding
# binary output images.
python resizer.py --input_base_path=[DATASET_PATH] --output_base_path=[OUTPUT_DATASET_PATH] --height=224 --width=224

# Train the model.
python train.py --epochs=140 --batch_size=24 --tr_dir=[OUTPUT_DATASET_PATH/training] --test_dir=[OUTPUT_DATASET_PATH/testing] --model_path=[.h5_FILE_PATH]

# Infer the predictions for input test images.
python inference.py --model_path=[.h5_FILE_PATH] --image_size=224 --input_dir=[OUTPUT_DATASET_PATH/testing] --output_dir=[PREDICTION_OUTPUT_DIR]