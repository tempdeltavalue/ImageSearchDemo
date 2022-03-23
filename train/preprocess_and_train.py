import glob
import os
from datetime import datetime

import tensorflow as tf

from train.triplet_model_container import TripletModelContainer
from train.triplet_generator import TripletGenerator


from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

target_shape = (224, 224)



def train(input_images_path,
          output_base_images_path,
          base_checkpoint_path,
          logs_path,
          epochs):

    if os.path.exists(base_checkpoint_path) is False:
        os.mkdir(base_checkpoint_path)

    input_img_paths = glob.glob(os.path.join(input_images_path, "*"))
    output_img_paths = glob.glob(os.path.join(output_base_images_path, "*"))

    len_inp_imgs = int(len(input_img_paths)*0.7)
    len_output_imgs = int(len(output_img_paths)*0.7)

    train_input_img_paths = input_img_paths[0:len_inp_imgs]
    train_output_img_paths = output_img_paths[0:len_output_imgs]

    triplet_generator = TripletGenerator(input_img_paths=train_input_img_paths,
                                         output_img_paths=train_output_img_paths)

    test_input_img_paths = input_img_paths[len_inp_imgs:]
    test_output_img_paths = output_img_paths[len_output_imgs:]

    test_triplet_generator = TripletGenerator(input_img_paths=test_input_img_paths,
                                              output_img_paths=test_output_img_paths)
    # strategy = tf.distribute.OneDeviceStrategy()
    #
    # with strategy.scope():
    model_container = TripletModelContainer()
    siamese_model = model_container.model

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    checkpoint = tf.train.Checkpoint(model=siamese_model)

    siamese_model.compile()

    date_string = datetime.now().strftime("%Y%m%d-%H%M%S")

    logdir = os.path.join(logs_path, date_string)
    file_writer = tf.summary.create_file_writer(logdir)
    file_writer.set_as_default()

    save_ch_path = os.path.join(base_checkpoint_path, date_string)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(triplet_generator):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.

                loss_value = _compute_loss(x_batch_train, siamese_model)

            grads = tape.gradient(loss_value, siamese_model.trainable_weights)

            optimizer.apply_gradients(zip(grads, siamese_model.trainable_weights))

        train_loss_value = tf.reduce_mean(loss_value)

        test_loss_value = 0
        for _, x_batch_test in enumerate(test_triplet_generator):
            test_loss_value += tf.reduce_mean(_compute_loss(x_batch_test, siamese_model))

        test_loss_value = test_loss_value / len(test_triplet_generator)
        tf.summary.scalar('train loss', data=train_loss_value, step=epoch)
        tf.summary.scalar('test loss', data=test_loss_value, step=epoch)

        e_save_ch_path = save_ch_path + "/e{}/".format(epoch)
        checkpoint.save(e_save_ch_path)

        print("Training loss at epoch %d: %.4f test loss %.4f" % (epoch,
                                                                  train_loss_value,
                                                                  test_loss_value))


def _compute_loss(data, model, margin=0.5):
    # The output of the network is a tuple containing the distances
    # between the anchor and the positive example, and the anchor and
    # the negative example.
    ap_distance, an_distance = model(data)

    # Computing the Triplet Loss by subtracting both distances and
    # making sure we don't get a negative value.
    loss = ap_distance - an_distance
    loss = tf.maximum(loss + margin, 0.0)
    return loss

if __name__ == "__main__":
    input_images_path = r"C:\Users\m\Desktop\IMTestData\input_images"
    output_base_images_path = r"C:\Users\m\Desktop\im_search_data"  # search data !
    base_checkpoint_path = r"C:\Users\m\Desktop\ImageSearch\train\checkpoints"
    logs_path = ""

    model_container = TripletModelContainer()
    siamese_model = model_container.model

    train(input_images_path, output_base_images_path, base_checkpoint_path, "logs", 10)

