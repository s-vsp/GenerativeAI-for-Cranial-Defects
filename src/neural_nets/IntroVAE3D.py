import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, models, metrics
import pickle

class Sampler(layers.Layer):
    """
    Input is of shape -> (None, latent_dim), where None represents the future batch size
    """
    def call(self, inputs):
        Z_mean, Z_log_var = inputs
        batch_size = tf.shape(Z_mean)[0]
        latent_dim = tf.shape(Z_mean)[1]
        
        # Use reparametrization trick
        epsilon = tf.random.normal(shape=(batch_size, latent_dim))
        Z = Z_mean + tf.exp(Z_log_var / 2) * epsilon
        return Z


class ResidualBlock(models.Model):
    def __init__(self, filters, kernel_size):
        super(ResidualBlock, self).__init__()

        self.block = models.Sequential(
            [
                layers.Conv3D(filters=filters, kernel_size=kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv3D(filters=filters, kernel_size=kernel_size, padding="same"),
                layers.BatchNormalization(),
                layers.LeakyReLU()
            ]
        )

        self.x = models.Sequential(
            [
                layers.Conv3D(filters=filters, kernel_size=kernel_size, padding="same")
            ]
        )
    
    def call(self, inputs):
        out = layers.Add()([self.block(inputs), self.x(inputs)])
        out = layers.BatchNormalization()(out)
        out = layers.LeakyReLU()(out)
        return out


def make_IntroVAE3D_inference_model(input_shape: int=128, latent_dim: int=256):
    inputs = layers.Input(shape=(input_shape,input_shape,input_shape,2), name="Input_layer")

    # Convolutional block 1
    block_1 = models.Sequential(
        [
            layers.Conv3D(16, (5,5,5), (1,1,1), padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.AveragePooling3D()
        ], name="Conv3D_block_1"
    )(inputs)

    # Residual Convolutional block 1
    res_block_1 = models.Sequential(
        [
            ResidualBlock(32, (1,1,1)),
            ResidualBlock(32, (3,3,3)),
            ResidualBlock(32, (3,3,3)),
            layers.AveragePooling3D()
        ], name="Residual_Conv3D_block_1"
    )(block_1)

    # Residual Convolutional block 2
    res_block_2 = models.Sequential(
        [
            ResidualBlock(64, (1,1,1)),
            ResidualBlock(64, (3,3,3)),
            ResidualBlock(64, (3,3,3)),
            layers.AveragePooling3D()
        ], name="Residual_Conv3D_block_2"
    )(res_block_1)

    # Residual Convolutional block 3
    res_block_3 = models.Sequential(
        [
            ResidualBlock(128, (1,1,1)),
            ResidualBlock(128, (3,3,3)),
            ResidualBlock(128, (3,3,3)),
            layers.AveragePooling3D()
        ], name="Residual_Conv3D_block_3"
    )(res_block_2)

    # Residual Convolutional block 4
    res_block_4 = models.Sequential(
        [
            ResidualBlock(256, (3,3,3)),
            ResidualBlock(256, (3,3,3)),
            layers.AveragePooling3D()
        ], name="Residual_Conv3D_block_4"
    )(res_block_3)

    # Residual Convolutional block 5
    res_block_5 = models.Sequential(
        [
            ResidualBlock(512, (3,3,3)),
            ResidualBlock(512, (3,3,3)),
        ], name="Residual_Conv3D_block_5"
    )(res_block_4)

    # Reshape
    reshape = models.Sequential(
        [
            layers.Reshape(target_shape=(1,1,1,512*4**3,))
        ]
    )(res_block_5)

    # Dense layer 1
    dense_1 = models.Sequential(
        [
            layers.Dense(latent_dim*2),
        ], name="Dense_layer_1"
    )(reshape)

    # Split tensor - compute mean and variance and sample Z
    flatten = layers.Flatten()(dense_1)
    Z_mean, Z_log_var = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(flatten)
    Z = Sampler()([Z_mean, Z_log_var])

    inference_model = models.Model(inputs=inputs, outputs=[Z_mean, Z_log_var, Z], name="Inference_model")
    return inference_model

def make_IntroVAE3D_generator(latent_dim: int=256):
    inputs = layers.Input(shape=(latent_dim,), name="Input_layer")

    # Dense layer 1
    dense_1 = models.Sequential(
        [
            layers.Reshape(target_shape=(1,1,1,latent_dim)),
            layers.Dense(512*4**3),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name="Dense_layer_1"
    )(inputs)

    # Residual Convolutional block 1
    res_block_1 = models.Sequential(
        [
            layers.Reshape(target_shape=(4,4,4,512)),
            ResidualBlock(512, (3,3,3)),
            ResidualBlock(512, (3,3,3))
        ], name="Residual_Conv3D_block_1"
    )(dense_1)

    # Residual Convolutional block 2
    res_block_2 = models.Sequential(
        [
            layers.UpSampling3D((2,2,2)),
            ResidualBlock(256, (1,1,1)),
            ResidualBlock(256, (3,3,3)),
            ResidualBlock(256, (3,3,3))
        ], name="Residual_Conv3D_block_2"
    )(res_block_1)

    # Residual Convolutional block 3
    res_block_3 = models.Sequential(
        [
            layers.UpSampling3D((2,2,2)),
            ResidualBlock(128, (1,1,1)),
            ResidualBlock(128, (3,3,3)),
            ResidualBlock(128, (3,3,3))
        ], name="Residual_Conv3D_block_3"
    )(res_block_2)

    # Residual Convolutional block 4
    res_block_4 = models.Sequential(
        [
            layers.UpSampling3D((2,2,2)),
            ResidualBlock(64, (1,1,1)),
            ResidualBlock(64, (3,3,3)),
            ResidualBlock(64, (3,3,3))
        ], name="Residual_Conv3D_block_4"
    )(res_block_3)

    # Residual Convolutional block 5
    res_block_5 = models.Sequential(
        [
            layers.UpSampling3D((2,2,2)),
            ResidualBlock(32, (1,1,1)),
            ResidualBlock(32, (3,3,3)),
            ResidualBlock(32, (3,3,3))
        ], name="Residual_Conv3D_block_5"
    )(res_block_4)

    # Residual Convolutional block 6
    res_block_6 = models.Sequential(
        [
            layers.UpSampling3D((2,2,2)),
            ResidualBlock(16, (1,1,1)),
            ResidualBlock(16, (3,3,3)),
            ResidualBlock(16, (3,3,3))
        ], name="Residual_Conv3D_block_6"
    )(res_block_5)

    # Output layer
    outputs = models.Sequential(
        [
            layers.Conv3D(3, (5,5,5), (1,1,1), padding="same")
        ]
    )(res_block_6)
    # TODO: Add activation function at the end

    generator = models.Model(inputs=inputs, outputs=outputs, name="Generator")
    return generator


def reconstruction_loss(data, reconstruction):
    return tf.reduce_mean(losses.mean_squared_error(data, reconstruction))


def kl_loss(Z_mean, Z_log_var):
    return tf.reduce_sum(-0.5 * (Z_log_var - tf.square(Z_mean) + 1 - tf.exp(Z_log_var)), axis=-1)


"""
Training details for hyperparameters tuning:
    Autors suggest to firstly perform a few training epochs of classic VAE so with alpha=0. Then based on the losses
    values mainly the m parameter is selected such that m > kl_loss (in model: inference_model_kl_loss). 
    Larger the beta is larger the m is.

    NOTE: During training the generator its standard kl_loss for Z is not monitored (following the training details 
    from the paper)
"""
class IntroVAE3D_model(models.Model):
    def __init__(self, inference_model, generator, latent_dim, alpha, beta, m):
        super(IntroVAE3D_model, self).__init__()
        self.inference_model = inference_model
        self.generator = generator
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        self.m = m

    def compile(self, inference_model_optimizer, generator_optimizer, reconstruction_loss, kl_loss):
        super(IntroVAE3D_model, self).compile()
        self.inference_model_optimizer = inference_model_optimizer
        self.generator_optimizer = generator_optimizer
        self.reconstruction_loss = reconstruction_loss
        self.kl_loss = kl_loss
        self.inference_model_reconstruction_loss_metric = metrics.Mean(name="inference_model_reconstruction_loss")
        self.inference_model_kl_loss_metric = metrics.Mean(name="inference_model_kl_loss")
        self.inference_model_total_loss_metric = metrics.Mean(name="inference_model_total_loss")
        self.generator_reconstruction_loss_metric = metrics.Mean(name="generator_reconstruction_loss")
        self.generator_total_loss_metric = metrics.Mean(name="generator_total_loss")

    @property
    def metrics(self):
        return [
            self.inference_model_reconstruction_loss_metric, 
            self.inference_model_kl_loss_metric, 
            self.inference_model_total_loss_metric,
            self.generator_reconstruction_loss_metric, 
            self.generator_total_loss_metric
        ]

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        Zp = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # Train the inference model
        with tf.GradientTape() as tape:
            Z_mean, Z_log_var, Z = self.inference_model(data)
            kl_loss = self.kl_loss(Z_mean, Z_log_var)
            Xr = self.generator(Z)
            Xp = self.generator(Zp)
            reconstruction_loss = self.reconstruction_loss(data, Xr)
            Zr_mean, Zr_log_var, Zr = self.inference_model(tf.stop_gradient(Xr))
            Zpp_mean, Zpp_log_var, Zpp = self.inference_model(tf.stop_gradient(Xp))
            inference_model_loss = kl_loss + self.alpha * (tf.maximum(0, self.m - self.kl_loss(Zr_mean, Zr_log_var)) + tf.maximum(0, self.m - self.kl_loss(Zpp_mean, Zpp_log_var))) + self.beta * reconstruction_loss
        grads_inference_model = tape.gradient(inference_model_loss, self.inference_model.trainable_variables)
        self.inference_model_optimizer.apply_gradients(zip(grads_inference_model, self.inference_model.trainable_variables))
        
        # Update inference model metrics
        self.inference_model_reconstruction_loss_metric.update_state(reconstruction_loss)
        self.inference_model_kl_loss_metric.update_state(kl_loss)
        self.inference_model_total_loss_metric.update_state(inference_model_loss)

        # Train the generator
        with tf.GradientTape() as tape:
            Z_mean, Z_log_var, Z = self.inference_model(data)
            Xr = self.generator(Z)
            Xp = self.generator(Zp)
            reconstruction_loss = self.reconstruction_loss(data, Xr)
            Zr_mean, Zr_log_var, Zr = self.inference_model(Xr)
            Zpp_mean, Zpp_log_var, Zpp = self.inference_model(Xp)
            generator_loss = self.alpha * (self.kl_loss(Zr_mean, Zr_log_var) + self.kl_loss(Zpp_mean, Zpp_log_var)) + self.beta * reconstruction_loss
        grads_generator = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads_generator, self.generator.trainable_variables))

        # Update generator metrics
        self.generator_reconstruction_loss_metric.update_state(reconstruction_loss)
        self.generator_total_loss_metric.update_state(generator_loss)

        return {
            "inference_model_reconstruction_loss": self.inference_model_reconstruction_loss_metric.result(),
            "inference_model_kl_loss": self.inference_model_kl_loss_metric.result(),
            "inference_model_total_loss": self.inference_model_total_loss_metric.result(),
            "generator_reconstruction_loss": self.generator_reconstruction_loss_metric.result(),
            "generator_total_loss": self.generator_total_loss_metric.result()
        }

if __name__ == "__main__":
    A = make_IntroVAE3D_inference_model(128)
    print(A.summary())
    B = make_IntroVAE3D_generator()
    print(B.summary())