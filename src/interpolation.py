import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import optimizers

from utils import save_to_nrrd, load_nrrd
from metrics import latent_space_interpolation
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

    latent_1 = pickle.load(open("vis_latent_1.pkl", "rb"))
    latent_2 = pickle.load(open("vis_latent_3.pkl", "rb"))

    latent_space_interpolation(decoder=model.generator, model_type="GAN", latent_1=latent_1, latent_2=latent_2, steps=20)

if __name__ == "__main__":
    run()