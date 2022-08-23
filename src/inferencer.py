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

from utils import save_to_nrrd
from neural_nets.WGANGP3D import make_WGANGP3D_critic, make_WGANGP3D_generator, critic_loss, generator_loss, gradient_penalty, WGANGP3D_model, WGANGP3D_monitor, threshold

def run():
    LATENT_DIM = 200
    HIDDEN_DIM = 512
    IMAGE_SHAPE = 128
    LR = 2e-4
    BETA_1 = 0.5
    BETA_2 = 0.9
    CRITIC_EXTRA_STEPS = 5
    LAMBDA_GP = 100

    generator = make_WGANGP3D_generator(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_shape=IMAGE_SHAPE)
    critic = make_WGANGP3D_critic(input_shape=IMAGE_SHAPE)

    model = WGANGP3D_model(generator=generator, critic=critic, latent_dim=LATENT_DIM, critic_extra_steps=CRITIC_EXTRA_STEPS, lambda_gp=LAMBDA_GP)

    model.compile(
        generator_optimizer=optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2),
        critic_optimizer=optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2),
        generator_loss_fn=generator_loss,
        critic_loss_fn=critic_loss
    )

    model.load_weights("data/WGANGP3D_data/checkpoints/WGANGP3D_training-0050-8th-run.ckpt").expect_partial()

    random_inputs = tf.random.uniform(shape=(1, LATENT_DIM))
    generated_skull = model.generator(random_inputs)
    skull = generated_skull.numpy()
    thresholded_skull = threshold(np.squeeze(skull))

    save_to_nrrd("data/WGANGP3D_data/example_skull_8th_run.nrrd", thresholded_skull, (0.1, 0.1, 0.1))

if __name__ == "__main__":
    run()

