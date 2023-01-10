import os
import re
import pathlib
import numpy as np
import SimpleITK as sitk
import io
import scipy.ndimage
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, models, metrics

from utils import create_dataset, concat_channels_0
from neural_nets.VAE3D import make_encoder, make_decoder, reconstruction_loss, kl_loss, VAE3D_model, VAE3D_monitor, LatentManifoldSaver

def run():
    LATENT_DIM = 200
    HIDDEN_DIM = 512
    IMAGE_SHAPE = 128
    LR = 1e-3
    BETA_1 = 0.9
    BETA_2 = 0.999
    BATCH_SIZE = 8
    NUM_EPOCHS = 30

    dataset = create_dataset(dataset_type=0)
    dataset = dataset.batch(BATCH_SIZE)

    encoder = make_encoder(input_shape=IMAGE_SHAPE, latent_dim=LATENT_DIM)
    decoder = make_decoder(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_shape=IMAGE_SHAPE)

    model = VAE3D_model(encoder=encoder, decoder=decoder, latent_dim=LATENT_DIM)

    # Load one skull for reconstruction visualization purposes - loading 1 same skull
    skull = concat_channels_0(tf.constant("./data/autoimplant/nrrd/defective_skull/bilateral/000.nrrd"))
    skull = tf.expand_dims(skull, axis=0)  

    # Load one skull for saving its representation in latent manifold - loading random skull
    skull_for_latents = next(iter(dataset.take(1)))[0]
    skull_for_latents = tf.expand_dims(skull_for_latents, axis=0)

    monitor = VAE3D_monitor("data/VAE3D_data/images/", data=skull)
    weights_track = keras.callbacks.ModelCheckpoint("data/VAE3D_data/checkpoints/VAE3D_training-{epoch:04d}.ckpt", save_weights_only=True, verbose=1)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="data/VAE3D_data/logs/", histogram_freq=1)
    latent_saver = LatentManifoldSaver("data/VAE3D_data/latent_manifold/", data=skull_for_latents)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2),
        reconstruction_loss_fn = reconstruction_loss,
        kl_loss_fn = kl_loss
    )

    model.save_weights("data/VAE3D_data/checkpoints/VAE3D_training-{epoch:04d}.ckpt".format(epoch=0))

    model.fit(dataset, epochs=NUM_EPOCHS, callbacks=[monitor, weights_track, tensorboard_callback, latent_saver])

if __name__ == "__main__":
    run()
