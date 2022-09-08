import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, models, metrics

"""
Implementation of the 3D Wasserstein GAN with gradient penalty architecture.
"""

def make_WGANGP3D_critic(input_shape: int=128):
    """
    Wasserstein GAN with gradient penalty critic function with shapes appropriate for the skull implants problem.

    Args:
        - input_shape: shape of the input image: input_shape x input_shape x input_shape x 2
    """
    inputs = layers.Input(shape=(input_shape,input_shape,input_shape,2), name="Input_layer")

    # Convolutional block 1
    block_1 = models.Sequential(
        [
            layers.Conv3D(64, (4,4,4), (2,2,2), padding="same"),
            layers.LayerNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_1"
    )(inputs)

    # Convolutional block 2
    block_2 = models.Sequential(
        [
            layers.Conv3D(128, (4,4,4), (2,2,2), padding="same"),
            layers.LayerNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_2"
    )(block_1)

    # Convolutional block 3
    block_3 = models.Sequential(
        [
            layers.Conv3D(256, (4,4,4), (2,2,2), padding="same"),
            layers.LayerNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_3"
    )(block_2)

    # Convolutional block 4
    block_4 = models.Sequential(
        [
            layers.Conv3D(512, (4,4,4), (2,2,2), padding="same"),
            layers.LayerNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_4"
    )(block_3)

    # Flatten layer
    flatten = models.Sequential(
        [
            layers.Flatten()
        ], name="Flatten"
    )(block_4)

    # Output layer
    outputs = models.Sequential(
        [
            layers.Dense(1)
        ], name="Output_layer"
    )(flatten)

    critic_model = models.Model(inputs=inputs, outputs=outputs, name="Critic")
    return critic_model


def make_WGANGP3D_generator(latent_dim: int=200, hidden_dim: int=512, output_shape: int=128):
    """
    Wasserstein GAN with gradient penalty generator function with shapes appropriate for the skull implants problem.

    Args:
        - latent_dim: dimensionality of noisy input
        - hidden_dim: equals to the penultimate number of channels in critic
        - output_shape: shape of the output image: output_shape x output_shape x output_shape x 2
    """
    inputs = layers.Input(shape=(latent_dim,), name="Input_layer")

    # Input dense layer
    input_dense = models.Sequential(
        [
            layers.Dense(int(output_shape / 2**4)**3 * hidden_dim)
        ], name="Input_dense"
    )(inputs)

    # Reshape
    reshape_layer = models.Sequential(
        [
            layers.Reshape((int(output_shape / 2**4), int(output_shape / 2**4), int(output_shape / 2**4), hidden_dim))
        ], name="Reshape_layer"
    )(input_dense)

     # Deconvolutional block 1
    block_1 = models.Sequential(
        [
            layers.Conv3DTranspose(512, (4,4,4), (1,1,1), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="DeConv3D_block_1"
    )(reshape_layer)

    # Deconvolutional block 2
    block_2 = models.Sequential(
        [
            layers.Conv3DTranspose(256, (4,4,4), (2,2,2), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="DeConv3D_block_2"
    )(block_1)

    # Deconvolutional block 3
    block_3 = models.Sequential(
        [
            layers.Conv3DTranspose(128, (4,4,4), (2,2,2), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="DeConv3D_block_3"
    )(block_2)

    # Deconvolutional block 4
    block_4 = models.Sequential(
        [
            layers.Conv3DTranspose(64, (4,4,4), (2,2,2), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="DeConv3D_block_4"
    )(block_3)

    # Output layer
    outputs = models.Sequential(
        [
            layers.Conv3DTranspose(2, (4,4,4), (2,2,2), padding="same", activation="tanh")
        ], name="Output_layer"
    )(block_4)

    generator_model = models.Model(inputs=inputs, outputs=outputs, name="Generator")
    return generator_model


def critic_loss(real_data, fake_data):
    return tf.reduce_mean(fake_data) - tf.reduce_mean(real_data)


def generator_loss(fake_data):
    return -tf.reduce_mean(fake_data)


def gradient_penalty(critic, real_data, fake_data):
    batch_size = tf.shape(real_data)[0]
    epsilon = tf.random.normal([batch_size, 1, 1, 1, 1], 0.0, 1.0)
    interpolated_data = real_data * epsilon + fake_data * (1 - epsilon)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_data)
        predictions = critic(interpolated_data)
    grads = tape.gradient(predictions, [interpolated_data])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]))
    gp = tf.reduce_mean((norm - 1.0)**2)
    return gp


class WGANGP3D_model(models.Model):
    def __init__(self, generator, critic, latent_dim, critic_extra_steps, lambda_gp):
        super(WGANGP3D_model, self).__init__()
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.critic_extra_steps = critic_extra_steps
        self.lambda_gp = lambda_gp


    def compile(self, generator_optimizer, critic_optimizer, generator_loss_fn, critic_loss_fn):
        super(WGANGP3D_model, self).compile()
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer
        self.generator_loss_fn = generator_loss_fn
        self.critic_loss_fn = critic_loss_fn
        self.generator_loss_metric = metrics.Mean(name="generator_loss")
        self.critic_loss_metric = metrics.Mean(name="critic_loss")

    @property
    def metrics(self):
        return [self.generator_loss_metric, self.critic_loss_metric]

    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]

        # Train the critic (more times than generator)
        for _ in range(self.critic_extra_steps):
            noise_input = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_data = self.generator(noise_input)
                fake_data_predictions = self.critic(fake_data)
                real_data_predictions = self.critic(real_data)
                gp = gradient_penalty(self.critic, real_data, fake_data)
                c_loss = self.critic_loss_fn(real_data_predictions, fake_data_predictions) + self.lambda_gp * gp
            grads_C = tape.gradient(c_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads_C, self.critic.trainable_variables))
        
        # Train the generator
        noise_data = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            predictions = self.critic(self.generator(noise_data))
            g_loss = self.generator_loss_fn(predictions)
        grads_G = tape.gradient(g_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads_G, self.generator.trainable_variables))

        # Monitor loss
        self.critic_loss_metric.update_state(c_loss)
        self.generator_loss_metric.update_state(g_loss)
        #return {"critic_loss": self.critic_loss_metric.result(), "generator_loss": self.generator_loss_metric.result()}
        return {"critic_loss": c_loss, "generator_loss": g_loss}
        

def threshold(image: np.array):
    thresholded = np.copy(image)
    thresholded[image < 0.5] = 0
    thresholded[image >= 0.5] = 1
    return thresholded


class WGANGP3D_monitor(keras.callbacks.Callback):
    def __init__(self, save_path, latent_dim=400):
        self.save_path = save_path
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_inputs = tf.random.uniform(shape=(1, self.latent_dim))
        generated_skull = self.model.generator(random_inputs)

        # Show only raw channels 
        skull = generated_skull.numpy()
        result1 = np.hstack([skull[0,:,:,40,0], skull[0,:,:,40,1]])
        result2 = np.hstack([np.rot90(skull[0,:,40,:,0]), np.rot90(skull[0,:,40,:,1])])
        result3 = np.hstack([np.rot90(skull[0,40,:,:,0]), np.rot90(skull[0,40,:,:,1])])
        result = np.vstack([result1, result2, result3])
        # For RGB-like purposes of keras arr-to-img visualization
        result_stacked = np.stack([result,result,result], axis=-1)

        image_channels = keras.preprocessing.image.array_to_img(result_stacked)
        image_channels.save(self.save_path + "generated_skull_on_epoch_{epoch}_8th_run.png".format(epoch=epoch))

        # Binary + Implant fitted
        thresholded_skull = threshold(skull)
        
        thresholded_skull_sum = thresholded_skull[0,:,:,:,0] + thresholded_skull[0,:,:,:,1]

        # Visualization purposes tricks
        thresholded_skull[thresholded_skull == 1] = 5
        thresholded_skull[thresholded_skull == 2] = 1
        thresholded_skull[thresholded_skull == 5] = 2
 
        thresholded_skull_sum[thresholded_skull_sum == 1] = 5
        thresholded_skull_sum[thresholded_skull_sum == 2] = 1
        thresholded_skull_sum[thresholded_skull_sum == 5] = 2

        result1_thresholded = np.hstack([thresholded_skull[0,:,:,40,0], thresholded_skull[0,:,:,40,1], thresholded_skull_sum[:,:,40]])
        result2_thresholded = np.hstack([np.rot90(thresholded_skull[0,:,40,:,0]), np.rot90(thresholded_skull[0,:,40,:,1]), np.rot90(thresholded_skull_sum[:,40,:])])
        result3_thresholded = np.hstack([np.rot90(thresholded_skull[0,40,:,:,0]), np.rot90(thresholded_skull[0,40,:,:,1]), np.rot90(thresholded_skull_sum[40,:,:])])
        result_thresholded = np.vstack([result1_thresholded, result2_thresholded, result3_thresholded])

        result_thresholded_stacked = np.stack([result_thresholded, result_thresholded, result_thresholded], -1)

        thresholded_channels = keras.preprocessing.image.array_to_img(result_thresholded_stacked)
        thresholded_channels.save(self.save_path + "generated_skull_thresholded_on_epoch_{epoch}_8th_run.png".format(epoch=epoch))
