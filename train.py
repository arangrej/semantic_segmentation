import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from optparse import OptionParser

from albumentations import (Compose, HorizontalFlip, RandomCrop, RandomRotate90, ShiftScaleRotate, Transpose, VerticalFlip)

from data_augmentation import DataGenerator
from unet_model import UNet

tf.logging.set_verbosity(tf.logging.ERROR)
image_size = 224


def aug_with_crop(image_size=image_size, crop_prob=1):

    return Compose([
        RandomCrop(width=image_size, height=image_size, p=crop_prob),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
    ], p=1)


def train(tr_dir, test_dir, model_path, epochs, batch_size):

    train_generator = DataGenerator(
                                    root_dir=tr_dir,
                                    image_folder='input/',
                                    mask_folder='output/',
                                    batch_size=batch_size,
                                    nb_y_features=1,
                                    augmentation=aug_with_crop,
                                    is_training=True
                                    )

    test_generator = DataGenerator(
                                    root_dir=test_dir,
                                    image_folder='input/',
                                    mask_folder='output/',
                                    batch_size=batch_size,
                                    nb_y_features=1,
                                    augmentation=None,
                                    is_training=True
                                    )

    mode_autosave = ModelCheckpoint(
                                    model_path,
                                    monitor='val_iou_score',
                                    mode='max',
                                    save_best_only=True,
                                    verbose=1,
                                    period=10
                                    )

    early_stopping = EarlyStopping(
                                    patience=10,
                                    verbose=1,
                                    mode='auto'
                                    )

    callbacks = [early_stopping, mode_autosave]

    model = UNet.get_unet(image_size)
    model.fit_generator(
                        train_generator,
                        shuffle=True,
                        epochs=epochs,
                        use_multiprocessing=False,
                        validation_data=test_generator,
                        verbose=1,
                        callbacks=callbacks
                        )


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-e", "--epochs", action='store', type='int', dest='epochs', default = 150)
    parser.add_option("-k", "--batch_size", action='store', type='int', dest='batch_size', default=24)
    parser.add_option("-i", "--tr_dir", action='store', type='str', dest='tr_dir')
    parser.add_option("-o", "--test_dir", action='store', type='str', dest='test_dir')
    parser.add_option("-m", "--model_path", action='store', type='str', dest='model_path',
                      default='./model/semantic_segmentation.h5')

    options, _ = parser.parse_args()
    train(options.tr_dir, options.test_dir, options.model_path, options.epochs, options.batch_size)
