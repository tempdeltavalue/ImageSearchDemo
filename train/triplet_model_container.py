from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import backend as K

target_shape = (224, 224)

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

class TripletModelContainer:
    def __init__(self):
        backbone = ResNet50(
            weights="imagenet", input_shape=target_shape + (3,), include_top=False
        )

        self.backbone = Model(backbone.input,
                              backbone.output, name="backbone")

        flatten = layers.Flatten()(self.backbone.output)
        dense1 = layers.Dense(512, activation="relu")(flatten)
        dense1 = layers.BatchNormalization()(dense1)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        output = layers.Dense(256)(dense2)

        self.embedding = Model(self.backbone.input, output, name="Embedding")

        trainable = False
        for index, layer in enumerate(self.backbone.layers):
            if index > len(self.backbone.layers) * 0.7:
                trainable = True

            layer.trainable = trainable

        anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
        positive_input = layers.Input(name="positive", shape=target_shape + (3,))
        negative_input = layers.Input(name="negative", shape=target_shape + (3,))

        distances = DistanceLayer()(
            self.embedding(preprocess_input(anchor_input)),
            self.embedding(preprocess_input(positive_input)),
            self.embedding(preprocess_input(negative_input)),
        )
        self.model = Model(
            inputs=[anchor_input, positive_input, negative_input], outputs=distances
        )

    def get_backbone(self):
        output = self.embedding.get_layer('conv5_block3_out')
        backbone = Model(self.embedding.input, output.output)
        return backbone


if __name__ == "__main__":
    model_container = TripletModelContainer()
    print(model_container.model.input_shape)
    model_container.get_backbone()
    # model_container.model.summary()
