import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, models, metrics
import tensorflow_addons as tfa


class Sampler(layers.Layer):
    """
    Input is of shape -> (None, latent_dim), where None represents the future batch size
    """
    def call(self, inputs):
        Z_mean, Z_var = inputs
        batch_size = tf.shape(Z_mean)[0]
        latent_dim = tf.shape( Z_mean)[1]
        
        # Use reparametrization trick
        epsilon = tf.random.normal(shape=(batch_size, latent_dim))
        Z = Z_mean + tf.exp(Z_var / 2) * epsilon
        return Z


def make_encoder(input_shape: int=128, latent_dim : int=200):
    inputs = layers.Input(shape=(input_shape,input_shape,input_shape,2))
    
    # Convolutional block 1
    block_1 = models.Sequential(
        [
            layers.Conv3D(64, (4,4,4), (2,2,2), padding="same"),
            tfa.layers.GroupNormalization(groups=16),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_1"
    )(inputs)

    # Convolutional block 2
    block_2 = models.Sequential(
        [
            layers.Conv3D(128, (4,4,4), (2,2,2), padding="same"),
            tfa.layers.GroupNormalization(groups=16),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_2"
    )(block_1)

    # Convolutional block 3
    block_3 = models.Sequential(
        [
            layers.Conv3D(256, (4,4,4), (2,2,2), padding="same"),
            tfa.layers.GroupNormalization(groups=16),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_3"
    )(block_2)

    # Convolutional block 4
    block_4 = models.Sequential(
        [
            layers.Conv3D(512, (4,4,4), (2,2,2), padding="same"),
            tfa.layers.GroupNormalization(groups=16),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_4"
    )(block_3)

    # Flatten layer
    flatten = models.Sequential(
        [
            layers.Flatten()
        ], name="Flatten"
    )(block_4)

    # Dense layer
    dense_1 = models.Sequential(
        [
            layers.Dense(4096),
            layers.LeakyReLU(alpha=0.2),
        ], name="Dense_1"
    )(flatten)

    # Compute mean and standard deviation and sample Z
    Z_mean = layers.Dense(latent_dim, name="Z_mean")(dense_1)
    Z_var = layers.Dense(latent_dim, name="Z_var")(dense_1)
    Z = Sampler()([Z_mean, Z_var])

    encoder = models.Model(inputs=inputs, outputs=[Z_mean, Z_var, Z], name="Encoder")
    return encoder

def make_decoder(latent_dim: int=200, hidden_dim: int=512, output_shape: int=128):
    inputs = layers.Input(shape=(latent_dim,))
    
    # Input dense layer
    input_dense = models.Sequential(
        [
        layers.Dense(int(output_shape / 2**4)**3 * hidden_dim)
        ], name="Dense_1"
    )(inputs)

    # Reshape layer
    reshape_layer = models.Sequential(
        [
            layers.Reshape((int(output_shape / 2**4), int(output_shape / 2**4), int(output_shape / 2**4), hidden_dim))
        ], name="Resahpe_layer"
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

    decoder = models.Model(inputs=inputs, outputs=outputs, name="Decoder")

    return decoder

def reconstruction_loss(data, reconstruction):
    return tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(losses.binary_crossentropy(data, reconstruction), axis=(1,2,3))))

def kl_loss(Z_mean, Z_var):
    return tf.reduce_sum(-0.5 * (Z_var - tf.square(Z_mean) + 1 - tf.exp(Z_var)), axis=-1)

class VAE3D_model(models.Model):
    def __init__(self, encoder, decoder, latent_dim):
        super(VAE3D_model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def compile(self, optimizer, reconstruction_loss_fn, kl_loss_fn):
        super(VAE3D_model, self).compile()
        self.optimizer = optimizer
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.kl_loss_fn = kl_loss_fn
        self.reconstruction_loss_metric = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_metric = metrics.Mean(name="kl_loss")
        self.total_loss_metric = metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.reconstruction_loss_metric, self.kl_loss_metric, self.total_loss_metric]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            Z_mean, Z_var, Z = self.encoder(data)
            reconstruction = self.decoder(Z)
            reconstruction_loss = self.reconstruction_loss_fn(data, reconstruction)
            kl_loss = self.kl_loss_fn(Z_mean, Z_var)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.reconstruction_loss_metric.update_state(reconstruction_loss)
        self.kl_loss_metric.update_state(kl_loss)
        self.total_loss_metric.update_state(total_loss)
        return {
            "total_loss": self.total_loss_metric.result(), 
            "reconstruction_loss": self.reconstruction_loss_metric.result(),
            "kl_loss": self.kl_loss_metric.result()
        }

class VAE3D_monitor(keras.callbacks.Callback):
    def __init__(self, save_path, data):
        self.save_path = save_path
        self.data = data
    
    def on_epoch_end(self, epoch, logs=None):
        _, _, Z = self.model.encoder(self.data)
        reconstruction = self.model.decoder(Z)
        
        # Show input data (skull)
        input_skull = self.data.numpy()
        input_skull_proj_1 = np.hstack([input_skull[0,:,:,40,0], input_skull[0,:,:,40,1]])
        input_skull_proj_2 = np.hstack([np.rot90(input_skull[0,:,40,:,0]), np.rot90(input_skull[0,:,40,:,1])])
        input_skull_proj_3 = np.hstack([np.rot90(input_skull[0,40,:,:,0]), np.rot90(input_skull[0,40,:,:,1])])
        input_skull_result = np.vstack([input_skull_proj_1, input_skull_proj_2, input_skull_proj_3])

        # Show reconstructed raw channels 
        skull = reconstruction.numpy()
        result1 = np.hstack([skull[0,:,:,40,0], skull[0,:,:,40,1]])
        result2 = np.hstack([np.rot90(skull[0,:,40,:,0]), np.rot90(skull[0,:,40,:,1])])
        result3 = np.hstack([np.rot90(skull[0,40,:,:,0]), np.rot90(skull[0,40,:,:,1])])
        result = np.vstack([result1, result2, result3])

        # Stack input and reconstruction
        final_result = np.hstack([input_skull_result, result])

        # For RGB-like purposes of keras arr-to-img visualization
        result_stacked = np.stack([final_result,final_result,final_result], axis=-1)

        image_channels = keras.preprocessing.image.array_to_img(result_stacked)
        image_channels.save(self.save_path + "input_skull_and_reconstruction_on_{epoch}.png".format(epoch=epoch))
