import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, models, metrics

from src.utils import load_nrrd, downsample
from src.neural_nets.AE3D import make_AE3D_encoder, make_AE3D_decoder, reconstruction_loss, AE3D_model, AE3D_monitor


def run():
    LATENT_DIM = 200
    HIDDEN_DIM = 512
    IMAGE_SHAPE = 128
    LR = 1e-3
    BETA_1 = 0.9
    BETA_2 = 0.999

    encoder = make_AE3D_encoder(input_shape=IMAGE_SHAPE, latent_dim=LATENT_DIM)
    decoder = make_AE3D_decoder(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_shape=IMAGE_SHAPE)

    model = AE3D_model(encoder=encoder, decoder=decoder, latent_dim=LATENT_DIM)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2),
        reconstruction_loss_fn = reconstruction_loss,
    )

    model.load_weights("data/AE3D_data/checkpoints/AE3D_training-0005-2nd-run.ckpt").expect_partial()

    latents_REAL = list()
    for root, dirs, files in os.walk("data/autoimplant/nrrd/defective_skull"):
        for file in files:
            volume1, _, _, _ = load_nrrd(root + "/" + file)
            volume2, _, _, _ = load_nrrd(root.replace("defective_skull", "implant") + "/" + file)

            volume1 = downsample(volume1, 4)
            volume2 = downsample(volume2, 4)

            volume1 = np.expand_dims(volume1, axis=-1)
            volume2 = np.expand_dims(volume2, axis=-1)
            volume = np.concatenate([volume1, volume2], axis=-1)
            
            volume = np.expand_dims(volume, axis=0)
            latent = model.encoder(volume)
            latents_REAL.append(latent.numpy().squeeze())
    
    array_latents_REAL = np.array(latents_REAL)
    np.save("data/latent_representations/REAL_latent", array_latents_REAL)

if __name__ == "__main__":
    run()
