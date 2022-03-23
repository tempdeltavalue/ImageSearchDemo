import os
import glob
import random

import numpy as np

import tensorflow as tf

target_shape = (224, 224)


class TripletGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 input_img_paths,
                 output_img_paths,
                 batch_size=16):

        'Initialization'
        splitter = "/"  # "\\"

        self.batch_size = batch_size

        output_img_paths = list(sorted(output_img_paths,
                                       key=lambda x: int(x.split(splitter)[-1])))

        self.anchor_img_paths = list(sorted(input_img_paths,
                                       key=lambda x: int(
                                           x.split(splitter)[-1].split(".")[0].split("_")[1])))  # 0_0 0_0 0_0

        self.pos_img_paths = []

        for index, _ in enumerate(input_img_paths):
            positive_path = output_img_paths[index]
            positive_img_path = list(filter(lambda x: ".png" in x, glob.glob(os.path.join(positive_path, "*"))))[0]

            self.pos_img_paths.append(positive_img_path)

        self.shuffle = True

        self.on_epoch_end()


    def preprocess_triplets(self, anchor, positive, negative):
        """
        Given the filenames corresponding to the three images, load and
        preprocess them.
        """
        return (
            self.preprocess_image(anchor),
            self.preprocess_image(positive),
            self.preprocess_image(negative),
        )

    def preprocess_image(self, filename):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
        """

        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, target_shape)
        return image


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.anchor_img_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        anc = np.empty((self.batch_size, 224, 224, 3))
        pos = np.empty((self.batch_size, 224, 224, 3))
        neg = np.empty((self.batch_size, 224, 224, 3))

        for index, k in enumerate(indexes):
            r = list(range(0,  k)) + list(range(k + 1, len(self.anchor_img_paths) - 1))

            negative_path = self.pos_img_paths[random.choice(r)]  # anything except current ind

            item = self.preprocess_triplets(self.anchor_img_paths[k],
                                            self.pos_img_paths[k],
                                            negative_path)
            anc[index] = item[0]
            pos[index] = item[1]
            neg[index] = item[2]

        return [anc, pos, neg]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.anchor_img_paths))

        if self.shuffle is True:
            np.random.shuffle(self.indexes)
