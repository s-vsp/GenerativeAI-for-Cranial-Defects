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
from neural_nets.IntroVAE3D import make_IntroVAE3D_inference_model, make_IntroVAE3D_generator, reconstruction_loss, kl_loss, IntroVAE3D_model, IntroVAE3D_ReconstructionMonitor, IntroVAE3D_NewSamplesMonitor

def run():
    LATENT_DIM = 256
    IMAGE_SHAPE = 128
    BATCH_SIZE = 8
    NUM_EPOCHS = 50

    # Optimizer hyperparams
    LR = 1e-3
    BETA_1 = 0.9
    BETA_2 = 0.999
    
    # Model hyperparams
    ALPHA = 0.25
    BETA = 1.0 
    M = 5.0

    dataset = create_dataset(dataset_type=0)
    dataset = dataset.batch(BATCH_SIZE)

    inference_model = make_IntroVAE3D_inference_model(input_shape=IMAGE_SHAPE, latent_dim=LATENT_DIM)
    generator = make_IntroVAE3D_generator(latent_dim=LATENT_DIM)

    keras.utils.plot_model(inference_model, "data/IntroVAE3D_data/inference_model.png", show_shapes=True)
    keras.utils.plot_model(generator, "data/IntroVAE3D_data/generator.png", show_shapes=True)

    model = IntroVAE3D_model(inference_model=inference_model, generator=generator, latent_dim=LATENT_DIM, alpha=ALPHA, beta=BETA, m=M)
 
    # Load one skull for reconstruction visualization purposes - loading 1 same skull
    skull = concat_channels_0(tf.constant("./data/autoimplant/nrrd/defective_skull/bilateral/000.nrrd"))
    skull = tf.expand_dims(skull, axis=0)

    reconstruction_monitor = IntroVAE3D_ReconstructionMonitor("data/IntroVAE3D_data/images/", data=skull)
    new_samples_monitor = IntroVAE3D_NewSamplesMonitor("data/IntroVAE3D_data/images/", latent_dim=LATENT_DIM, batch_size=BATCH_SIZE)
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

    model.fit(dataset, epochs=NUM_EPOCHS, callbacks=[weights_track, tensorboard_callback, reconstruction_monitor, new_samples_monitor])

if __name__ == "__main__":
    run()

