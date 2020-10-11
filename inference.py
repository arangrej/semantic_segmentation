import tensorflow as tf
import numpy as np
import os
from skimage.io import imsave
from optparse import OptionParser
from data_augmentation import DataGenerator
from unet_model import UNet

tf.logging.set_verbosity(tf.logging.ERROR)

batch_size = 1


def infer(model_path, image_size, input_dir, output_dir):

    test_generator = DataGenerator(
                                    root_dir=input_dir,
                                    image_folder='input/',
                                    mask_folder='output/',
                                    batch_size=batch_size,
                                    nb_y_features=1,
                                    augmentation=None,
                                    shuffle=False
                                    )

    model = UNet.get_unet(image_size)
    model.load_weights(model_path)

    num_images = len(os.listdir(os.path.join(input_dir, 'input')))

    for i in range(num_images):
        x_test, y_test, image_name = test_generator.__getitem__(i)
        predicted = model.predict(np.expand_dims(x_test[0], axis=0)).reshape(image_size, image_size)
        imsave(os.path.join(output_dir, image_name[0]), predicted)


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-m", "--model_path", action='store', type='str', dest='model_path', default='./model/semantic_segmentation.h5')
    parser.add_option("-n", "--image_size", action='store', type='int', dest='image_size', default=224)
    parser.add_option("-i", "--input_dir", action='store', type='str', dest='input_dir')
    parser.add_option("-o", "--output_dir", action='store', type='str', dest='output_dir')

    options, _ = parser.parse_args()
    infer(options.model_path, options.image_size, options.input_dir, options.output_dir)
