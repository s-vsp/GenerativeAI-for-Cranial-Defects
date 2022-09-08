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
from neural_nets.VAE_WGANGP3D import make_VAE_WGANGP3D_critic, make_VAE_WGANGP3D_generator, critic_loss, generator_loss, gradient_penalty, get_latent_vector, VAE_WGANGP3D_model, VAE_WGANGP3D_monitor

def run():
    LATENT_DIM = 200
    HIDDEN_DIM = 512
    IMAGE_SHAPE = 128
    LR = 2e-4
    BETA_1 = 0.5
    BETA_2 = 0.9
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    CRITIC_EXTRA_STEPS = 5
    LAMBDA_GP = 100

    dataset = create_dataset(dataset_type=0)
    dataset = dataset.batch(BATCH_SIZE)

    generator = make_VAE_WGANGP3D_generator(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_shape=IMAGE_SHAPE)
    critic = make_VAE_WGANGP3D_critic(input_shape=IMAGE_SHAPE)

    model = VAE_WGANGP3D_model(
        generator=generator, critic=critic, critic_extra_steps=CRITIC_EXTRA_STEPS, lambda_gp=LAMBDA_GP, 
        Z_mean_path="data/VAE3D_data/latent_manifold/Z_mean_on_epoch_29.pkl", Z_var_path="data/VAE3D_data/latent_manifold/Z_var_on_epoch_29.pkl"
    )

    monitor = VAE_WGANGP3D_monitor(
        save_path="data/VAE_WGANGP3D_data/images/", Z_mean_path="data/VAE3D_data/latent_manifold/Z_mean_on_epoch_29.pkl", 
        Z_var_path="data/VAE3D_data/latent_manifold/Z_var_on_epoch_29.pkl"
    )
    weights_track = keras.callbacks.ModelCheckpoint("data/VAE_WGANGP3D_data/checkpoints/VAE_WGANGP3D_training-{epoch:04d}.ckpt", save_weights_only=True, verbose=1)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="data/VAE_WGANGP3D_data/logs/", histogram_freq=1)

    model.compile(
        generator_optimizer=optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2),
        critic_optimizer=optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2),
        generator_loss_fn=generator_loss,
        critic_loss_fn=critic_loss
    )

    #model.load_weights("data/WGANGP3D_data/checkpoints/WGANGP3D_training-0050.ckpt").expect_partial()
    model.save_weights("data/VAE_WGANGP3D_data/checkpoints/VAE_WGANGP3D_training-{epoch:04d}.ckpt".format(epoch=0))

    model.fit(dataset, epochs=NUM_EPOCHS, callbacks=[monitor, weights_track, tensorboard_callback])

if __name__ == "__main__":
    run()
