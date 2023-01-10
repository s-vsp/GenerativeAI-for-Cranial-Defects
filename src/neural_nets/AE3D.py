import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, models, metrics
import tensorflow_addons as tfa
import pickle


def make_AE3D_encoder(input_shape: int=128, latent_dim : int=200):
    inputs = layers.Input(shape=(input_shape,input_shape,input_shape,2))
    
    # Convolutional block 1
    block_1 = models.Sequential(
        [
            layers.Conv3D(64, (4,4,4), (2,2,2), padding="same"),
            tfa.layers.GroupNormalization(groups=64),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_1"
    )(inputs)

    # Convolutional block 2
    block_2 = models.Sequential(
        [
            layers.Conv3D(128, (4,4,4), (2,2,2), padding="same"),
            tfa.layers.GroupNormalization(groups=128),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_2"
    )(block_1)

    # Convolutional block 3
    block_3 = models.Sequential(
        [
            layers.Conv3D(256, (4,4,4), (2,2,2), padding="same"),
            tfa.layers.GroupNormalization(groups=256),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_3"
    )(block_2)

    # Convolutional block 4
    block_4 = models.Sequential(
        [
            layers.Conv3D(512, (4,4,4), (2,2,2), padding="same"),
            tfa.layers.GroupNormalization(groups=512),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_4"
    )(block_3)

    # Adaptive Average Pooling layer
    adp_avg_pool_1 = models.Sequential(
    [
        tfa.layers.AdaptiveAveragePooling3D((1,1,1))
    ]
    )(block_4)

    # Flatten layer
    flatten = models.Sequential(
        [
            layers.Flatten()
        ], name="Flatten"
    )(adp_avg_pool_1)

    # Compute mean and standard deviation and sample Z
    Z = layers.Dense(latent_dim, name="Latent_dim")(flatten)

    encoder = models.Model(inputs=inputs, outputs=Z, name="Encoder")
    return encoder


def make_AE3D_decoder(latent_dim: int=200, hidden_dim: int=512, output_shape: int=128):
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
            tfa.layers.GroupNormalization(groups=512),
            layers.ReLU()
        ], name="DeConv3D_block_1"
    )(reshape_layer)

    # Deconvolutional block 2
    block_2 = models.Sequential(
        [
            layers.Conv3DTranspose(256, (4,4,4), (2,2,2), padding="same"),
            tfa.layers.GroupNormalization(groups=256),
            layers.ReLU()
        ], name="DeConv3D_block_2"
    )(block_1)

    # Deconvolutional block 3
    block_3 = models.Sequential(
        [
            layers.Conv3DTranspose(128, (4,4,4), (2,2,2), padding="same"),
            tfa.layers.GroupNormalization(groups=128),
            layers.ReLU()
        ], name="DeConv3D_block_3"
    )(block_2)

    # Deconvolutional block 4
    block_4 = models.Sequential(
        [
            layers.Conv3DTranspose(64, (4,4,4), (2,2,2), padding="same"),
            tfa.layers.GroupNormalization(groups=64),
            layers.ReLU()
        ], name="DeConv3D_block_4"
    )(block_3)

    # Output layer
    outputs = models.Sequential(
        [
            layers.Conv3DTranspose(2, (4,4,4), (2,2,2), padding="same", activation="sigmoid")
        ], name="Output_layer"
    )(block_4)

    decoder = models.Model(inputs=inputs, outputs=outputs, name="Decoder")

    return decoder


def reconstruction_loss(data, reconstruction):
    return tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(losses.binary_crossentropy(data, reconstruction))))


class AE3D_model(models.Model):
    def __init__(self, encoder, decoder, latent_dim):
        super(AE3D_model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def compile(self, optimizer, reconstruction_loss_fn):
        super(AE3D_model, self).compile()
        self.optimizer = optimizer
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.reconstruction_loss_metric = metrics.Mean(name="reconstruction_loss")

    @property
    def metrics(self):
        return [self.reconstruction_loss_metric]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            Z = self.encoder(data)
            reconstruction = self.decoder(Z)
            reconstruction_loss = self.reconstruction_loss_fn(data, reconstruction)
        grads = tape.gradient(reconstruction_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.reconstruction_loss_metric.update_state(reconstruction_loss)
        return {"reconstruction_loss": self.reconstruction_loss_metric.result()}


def threshold(image: np.array):
    thresholded = np.copy(image)
    thresholded[image < 0.5] = 0
    thresholded[image >= 0.5] = 1
    return thresholded


class AE3D_monitor(keras.callbacks.Callback):
    def __init__(self, save_path, data):
        self.save_path = save_path
        self.data = data
    
    def on_epoch_end(self, epoch, logs=None):
        Z = self.model.encoder(self.data)
        skull = self.model.decoder.predict(Z)
        
        # Show input data (skull)
        input_skull = self.data.numpy()
        input_skull_proj_1 = np.hstack([input_skull[0,:,:,40,0], input_skull[0,:,:,40,1]])
        input_skull_proj_2 = np.hstack([np.rot90(input_skull[0,:,40,:,0]), np.rot90(input_skull[0,:,40,:,1])])
        input_skull_proj_3 = np.hstack([np.rot90(input_skull[0,40,:,:,0]), np.rot90(input_skull[0,40,:,:,1])])
        input_skull_result = np.vstack([input_skull_proj_1, input_skull_proj_2, input_skull_proj_3])

        # Show reconstructed raw channels 
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