import os
import re
import pathlib
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import io
import scipy.ndimage
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, models, metrics

from utils import create_dataset
from neural_nets.DCGAN3D import make_DCGAN3D_discriminator, make_DCGAN3D_generator, DCGAN3D_model, DCGAN3D_monitor

def run():
    LATENT_DIM = 400
    HIDDEN_DIM = 512
    IMAGE_SHAPE = 128
    GENERATOR_LR = 2.5 * 1e-3
    DISCRIMINATOR_LR = 1e-5
    BETA_1 = 0.5
    BATCH_SIZE = 16
    NUM_EPOCHS = 10

    dataset = create_dataset(dataset_type=0)
    dataset = dataset.batch(BATCH_SIZE)

    generator = make_DCGAN3D_generator(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_shape=IMAGE_SHAPE)
    discriminator = make_DCGAN3D_discriminator(input_shape=IMAGE_SHAPE)
    
    model = DCGAN3D_model(generator=generator, discriminator=discriminator, latent_dim=LATENT_DIM)
    
    monitor = DCGAN3D_monitor("data/DCGAN3D_data/images/", latent_dim=LATENT_DIM)
    weights_track = keras.callbacks.ModelCheckpoint("data/DCGAN3D_data/checkpoints/DCGAN3D_training-{epoch:04d}.ckpt", save_weights_only=True, save_best_only=False)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="data/DCGAN3D_data/logs/", histogram_freq=1)    

    model.compile(
        generator_optimizer=optimizers.Adam(learning_rate=GENERATOR_LR, beta_1=BETA_1),
        discriminator_optimizer=optimizers.Adam(learning_rate=DISCRIMINATOR_LR, beta_1=BETA_1),
        generator_loss_fn=losses.BinaryCrossentropy(from_logits=True),
        discriminator_loss_fn=losses.BinaryCrossentropy(from_logits=True)
    )

    model.fit(dataset, epochs=NUM_EPOCHS, callbacks=[monitor, tensorboard_callback])
    print("finished...")

if __name__ == "__main__":
    run()
