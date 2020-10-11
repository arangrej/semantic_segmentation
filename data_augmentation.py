import os
import numpy as np
from keras.utils import Sequence
from sklearn.utils import shuffle
from skimage.io import imread
from albumentations import Resize


class DataGenerator(Sequence):

    def __init__(self, root_dir, image_folder='input/', mask_folder='output/',
                 batch_size=1, image_size=224, nb_y_features=1, augmentation=None, shuffle=True,
                 is_training=False):

        self.root_dir = root_dir
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_filenames = os.listdir(os.path.join(root_dir, image_folder))
        self.mask_names = os.listdir(os.path.join(root_dir, mask_folder))
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.image_size = image_size
        self.nb_y_features = nb_y_features
        self.indexes = None
        self.shuffle = shuffle
        self.is_training = is_training

    def __len__(self):
        """
        Calculates size of batch
        """
        return int(np.ceil(len(self.image_filenames) / (self.batch_size)))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle == True:
            self.image_filenames, self.mask_names = shuffle(self.image_filenames, self.mask_names)

    def read_image_mask(self, image_name, mask_name):
        return imread(os.path.join(self.root_dir, self.image_folder, image_name)) / 255, \
               (imread(os.path.join(self.root_dir, self.mask_folder, mask_name), as_gray=True) > 0).astype(np.int8), \
                image_name

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        data_index_min = int(index * self.batch_size)

        data_index_max = int(min((index + 1) * self.batch_size, len(self.image_filenames)))

        indexes = self.image_filenames[data_index_min:data_index_max]

        this_batch_size = len(indexes)  # The last batch can be smaller than the others

        # Defining dataset
        x = np.empty((this_batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        y = np.empty((this_batch_size, self.image_size, self.image_size, self.nb_y_features), dtype=np.uint8)
        nm = np.empty((this_batch_size), dtype=object)

        for i, sample_index in enumerate(indexes):

            x_sample, y_sample, image_name = self.read_image_mask(self.image_filenames[index * self.batch_size + i],
                                                      self.mask_names[index * self.batch_size + i])

            if self.augmentation is not None:
                augmented = self.augmentation(self.image_size)(image=x_sample, mask=y_sample)
                image_augm = augmented['image']
                mask_augm = augmented['mask'].reshape(self.image_size, self.image_size, self.nb_y_features)
                x[i, ...] = np.clip(image_augm, a_min=0, a_max=1)
                y[i, ...] = mask_augm
                nm[i] = image_name
            else:
                x_sample, y_sample, image_name = self.read_image_mask(self.image_filenames[index * 1 + i],
                                                          self.mask_names[index * 1 + i])
                augmented = Resize(height=(x_sample.shape[0] // 32) * 32, width=(x_sample.shape[1] // 32) * 32)(
                    image=x_sample, mask=y_sample)
                x_sample, y_sample = augmented['image'], augmented['mask']
                x[i, ...] = x_sample.reshape(1, x_sample.shape[0], x_sample.shape[1], 3).astype(np.float32)
                y[i, ...] = y_sample.reshape(1, x_sample.shape[0], x_sample.shape[1], self.nb_y_features).astype(np.uint8)
                nm[i] = image_name

        if self.is_training:
            return x, y
        else:
            return x, y, nm
