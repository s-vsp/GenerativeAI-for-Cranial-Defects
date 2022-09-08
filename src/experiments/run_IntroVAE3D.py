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

from utils import create_dataset
from neural_nets.IntroVAE3D import make_IntroVAE3D_inference_model, make_IntroVAE3D_generator, reconstruction_loss, kl_loss, IntroVAE3D_model

def run():
    LATENT_DIM = 256
    IMAGE_SHAPE = 128
    BATCH_SIZE = 8
    NUM_EPOCHS = 50

    # Optimizer hyperparams
    LR = 2e-4
    BETA_1 = 0.9
    BETA_2 = 0.999
    
    # Model hyperparams
    ALPHA = 0 # for classic VAE mannes
    BETA = 0.5
    M = 1000

    dataset = create_dataset(dataset_type=0)
    dataset = dataset.batch(BATCH_SIZE)

    inference_model = make_IntroVAE3D_inference_model(input_shape=IMAGE_SHAPE, latent_dim=LATENT_DIM)
    generator = make_IntroVAE3D_generator(latent_dim=LATENT_DIM)

    model = IntroVAE3D_model(inference_model=inference_model, generator=generator, latent_dim=LATENT_DIM, alpha=ALPHA, beta=BETA, m=M)
 
    #TODO: monitor = IntroVAE3D_monitor("data/IntroVAE3D_data/images/", latent_dim=LATENT_DIM)
    weights_track = keras.callbacks.ModelCheckpoint("data/IntroVAE3D_data/checkpoints/IntroVAE3D_training-{epoch:04d}.ckpt", save_weights_only=True, verbose=1)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="data/IntroVAE3D_data/logs/", histogram_freq=1)

    model.compile(
        inference_model_optimizer=optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2),
        generator_optimizer=optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2),
        reconstruction_loss=reconstruction_loss,
        kl_loss=kl_loss
    )

    #model.load_weights("data/IntroVAE3D_data/checkpoints/IntroVAE3D_training-0050.ckpt").expect_partial()
    model.save_weights("data/IntroVAE3D_data/checkpoints/IntroVAE3D_training-{epoch:04d}.ckpt".format(epoch=0))

    model.fit(dataset, epochs=NUM_EPOCHS, callbacks=[weights_track, tensorboard_callback])

if __name__ == "__main__":
    run()
