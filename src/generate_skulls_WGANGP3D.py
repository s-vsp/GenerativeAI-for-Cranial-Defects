import os
import re
import cc3d
import pathlib
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, models, metrics

from utils import load_nrrd, save_to_nrrd, extract_channels
from neural_nets.WGANGP3D import make_WGANGP3D_critic, make_WGANGP3D_generator, critic_loss, generator_loss, gradient_penalty, WGANGP3D_model, WGANGP3D_monitor, threshold

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

    
    noise = tf.random.normal(shape=(2, LATENT_DIM))
    latent_1 = noise[0]
    latent_2 = noise[1]
    plt.figure()
    plt.plot(latent_1, color='cornflowerblue')
    plt.plot(latent_2, color='indigo')
    plt.show()
    pickle.dump(latent_1, open('vis_latent_1.pkl', 'wb'))
    pickle.dump(latent_2, open('vis_latent_2.pkl', 'wb'))
    print(noise.shape)
    skulls = model.generator.predict(noise)
    thresholded_skulls = threshold(np.squeeze(skulls))
    thresholded_skull_1 = thresholded_skulls[0]
    thresholded_skull_2 = thresholded_skulls[1]

    defected_skull_1 = thresholded_skull_1[:,:,:,0]
    implant_1 = thresholded_skull_1[:,:,:,1]

    defected_skull_2 = thresholded_skull_2[:,:,:,0]
    implant_2 = thresholded_skull_2[:,:,:,1]

    # Postprocessing - 1. LOGICAL OPERATIONS
    implant_1 = np.logical_xor(implant_1, np.logical_and(implant_1, defected_skull_1))
    implant_2 = np.logical_xor(implant_2, np.logical_and(implant_2, defected_skull_2))

    # Postprocessing - 2. Connected Components Analysis
    defected_skull_1 = cc3d.connected_components(defected_skull_1, connectivity=6)
    implant_1  = cc3d.connected_components(implant_1, connectivity=6)

    defected_skull_2 = cc3d.connected_components(defected_skull_2, connectivity=6)
    implant_2  = cc3d.connected_components(implant_2, connectivity=6)

    defected_skull_1 = cc3d.dust(defected_skull_1, threshold=100, connectivity=6, in_place=False)
    defected_skull_1[defected_skull_1 != 0] = 1

    defected_skull_2 = cc3d.dust(defected_skull_2, threshold=100, connectivity=6, in_place=False)
    defected_skull_2[defected_skull_2 != 0] = 1

    implant_1 = cc3d.dust(implant_1, threshold=100, connectivity=6, in_place=False)
    implant_1[implant_1 != 0] = 1

    implant_2 = cc3d.dust(implant_2, threshold=100, connectivity=6, in_place=False)
    implant_2[implant_2 != 0] = 1

    save_to_nrrd("./visualization_defected_skull_1.nrrd", defected_skull_1, (0.1, 0.1, 0.1))
    save_to_nrrd("./visualization_implant_1.nrrd", implant_1, (0.1, 0.1, 0.1))
    save_to_nrrd("./visualization_defected_skull_2.nrrd", defected_skull_2, (0.1, 0.1, 0.1))
    save_to_nrrd("./visualization_implant_2.nrrd", implant_2, (0.1, 0.1, 0.1))

if __name__ == "__main__":
    run()
