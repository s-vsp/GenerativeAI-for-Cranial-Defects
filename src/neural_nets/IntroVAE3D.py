import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, models, metrics
import pickle


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


def make_IntroVAE3D_inference_model(input_shape: int=128, latent_dim : int=200):
    inputs = layers.Input(shape=(input_shape,input_shape,input_shape,2), name="Input_layer")
    
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
    ], name="Adaptive_Average_Pooling_1"
    )(block_4)

    # Flatten layer
    flatten = models.Sequential(
        [
            layers.Flatten()
        ], name="Flatten"
    )(adp_avg_pool_1)

    # Compute mean and variance and sample Z
    Z_mean = layers.Dense(latent_dim, name="Z_mean")(flatten)
    Z_var = layers.Dense(latent_dim, name="Z_var")(flatten)
    Z = Sampler()([Z_mean, Z_var])

    inference_model = models.Model(inputs=inputs, outputs=[Z_mean, Z_var, Z], name="Inference_model")
    return inference_model


def make_IntroVAE3D_generator(latent_dim: int=200, hidden_dim: int=512, output_shape: int=128):
    inputs = layers.Input(shape=(latent_dim,), name="Input_layer")
    
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
            layers.Conv3DTranspose(2, (4,4,4), (2,2,2), padding="same")
        ], name="Output_layer"
    )(block_4)

    generator = models.Model(inputs=inputs, outputs=outputs, name="Generator")
    return generator


def reconstruction_loss(data, reconstruction):
    return tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(tf.square(data - reconstruction), axis=0), axis=[1,2,3]))


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
            inference_model_loss = kl_loss + self.alpha * (tf.maximum(0.0, self.m - self.kl_loss(Zr_mean, Zr_log_var)) + tf.maximum(0.0, self.m - self.kl_loss(Zpp_mean, Zpp_log_var))) + self.beta * reconstruction_loss
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


def threshold(image: np.array):
    thresholded = np.copy(image)
    thresholded[image < 0.5] = 0
    thresholded[image >= 0.5] = 1
    return thresholded


class IntroVAE3D_ReconstructionMonitor(keras.callbacks.Callback):
    def __init__(self, save_path, data):
        self.save_path = save_path
        self.data = data

    def on_epoch_end(self, epoch, logs=None):
        _, _, Z = self.model.inference_model(self.data)
        reconstruction = self.model.generator(Z)

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


class IntroVAE3D_NewSamplesMonitor(keras.callbacks.Callback):
    def __init__(self, save_path, latent_dim, batch_size):
        self.save_path = save_path
        self.latent_dim = latent_dim
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        Zp = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        Xp = self.model.generator(Zp)

        # Show generated skull raw channels 
        skull = Xp.numpy()
        result1 = np.hstack([skull[0,:,:,40,0], skull[0,:,:,40,1]])
        result2 = np.hstack([np.rot90(skull[0,:,40,:,0]), np.rot90(skull[0,:,40,:,1])])
        result3 = np.hstack([np.rot90(skull[0,40,:,:,0]), np.rot90(skull[0,40,:,:,1])])
        result = np.vstack([result1, result2, result3])

        # For RGB-like purposes of keras arr-to-img visualization
        result_stacked = np.stack([result,result,result], axis=-1)

        image_channels = keras.preprocessing.image.array_to_img(result_stacked)
        image_channels.save(self.save_path + "generated_skull_on_epoch_{epoch}.png".format(epoch=epoch))

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
        thresholded_channels.save(self.save_path + "generated_skull_thresholded_on_epoch_{epoch}.png".format(epoch=epoch))


"""
Here in generating new samples we generate them from nose (normal dist.) and outputs should be nice.
In opposite VAE-WGAN had to use reparametrized version of noise to make nice outputs and when it was
using only normal dist. noise (without reparametrization) outputs where bad (terrible)
"""

if __name__ == "__main__":
    inferencer = make_IntroVAE3D_inference_model()
    generator = make_IntroVAE3D_generator()
    keras.utils.plot_model(inferencer, "INTROVAE_INF.png", show_shapes=True)
    keras.utils.plot_model(generator, "INTROVAE_GEN.png", show_shapes=True)
