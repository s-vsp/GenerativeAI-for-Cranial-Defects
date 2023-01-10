import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, models, metrics

#from utils import threshold

"""
Implementation of the basic DCGAN3D architecture, ref: [https://arxiv.org/pdf/1610.07584.pdf]
"""

def make_DCGAN3D_discriminator(input_shape: int=128):
    """
    3D-GAN discriminator function based on the architecture proposed in the original paper,
    but with shapes appropriate for the skull implants problem.

    Args:
        - input_shape: shape of the input image: input_shape x input_shape x input_shape x 2
    """
    inputs = layers.Input(shape=(input_shape,input_shape,input_shape,2), name="Input_layer")

    # Convolutional block 1
    block_1 = models.Sequential(
        [
            layers.Conv3D(64, (4,4,4), (2,2,2), padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_1"
    )(inputs)

    # Convolutional block 2
    block_2 = models.Sequential(
        [
            layers.Conv3D(128, (4,4,4), (2,2,2), padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_2"
    )(block_1)

    # Convolutional block 3
    block_3 = models.Sequential(
        [
            layers.Conv3D(256, (4,4,4), (2,2,2), padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ], name="Conv3D_block_3"
    )(block_2)

    # Convolutional block 4
    block_4 = models.Sequential(
        [
            layers.Conv3D(512, (4,4,4), (2,2,2), padding="same"),
            layers.BatchNormalization(),
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
            layers.Dense(1, activation="sigmoid")
        ], name="Output_layer"
    )(flatten)

    discriminator_model = models.Model(inputs=inputs, outputs=outputs, name="Discriminator")
    return discriminator_model


def make_DCGAN3D_generator(latent_dim: int=200, hidden_dim: int=512, output_shape: int=128):
    """
    3D-GAN generator function based on the architecture proposed in the original paper,
    but with shapes appropriate for the skull implants problem.

    Args:
        - latent_dim: dimensionality of noisy input
        - hidden_dim: equals to the penultimate number of channels in discriminator
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


def generator_loss(labels, predictions):
    return tf.reduce_sum(losses.binary_crossentropy(labels, predictions, from_logits=True))


def discriminator_loss(labels, predictions):
    return tf.reduce_sum(losses.binary_crossentropy(labels, predictions, from_logits=True))


class DCGAN3D_model(models.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(DCGAN3D_model, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
    
    def compile(self, generator_optimizer, discriminator_optimizer, generator_loss_fn, discriminator_loss_fn):
        super(DCGAN3D_model, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss_fn = generator_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn
        self.generator_loss_metric = metrics.Mean(name="generator_loss")
        self.discriminator_loss_metric = metrics.Mean(name="discriminator_loss")
    
    @property
    def metrics(self):
        return [self.generator_loss_metric, self.discriminator_loss_metric]

    def train_step(self, real_data):
        # Generate fake data
        batch_size = tf.shape(real_data)[0]
        noise_input = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_data = self.generator(noise_input)

        # Assign labels
        real_labels = tf.zeros(shape=(batch_size, 1))
        fake_labels = tf.ones(shape=(batch_size, 1))
        
        # Concatenate data and labels
        data = tf.concat([fake_data, real_data], axis=0)
        labels = tf.concat([fake_labels, real_labels], axis=0)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(data)
            discriminator_loss = self.discriminator_loss_fn(labels, predictions)
        grads_D = tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads_D, self.discriminator.trainable_variables))

        # Noise data to generator
        noise_data = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Misleadning labels
        misleading_labels = tf.zeros(shape=(batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(noise_data))
            generator_loss = self.generator_loss_fn(misleading_labels, predictions)
        grads_G = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads_G, self.generator.trainable_variables))

        # Monitor loss
        self.generator_loss_metric.update_state(generator_loss)
        self.discriminator_loss_metric.update_state(discriminator_loss)

        return {
            "Generator_loss": self.generator_loss_metric.result(),
            "Discriminator_loss": self.discriminator_loss_metric.result()
            }


class DCGAN3D_monitor(keras.callbacks.Callback):
    def __init__(self, save_path, latent_dim=400):
        self.save_path = save_path
        self.latent_dim = latent_dim
    
    def on_epoch_end(self, epoch, logs=None):
        random_inputs = tf.random.uniform(shape=(1, self.latent_dim))
        generated_skull = self.model.generator(random_inputs)
        
        skull = generated_skull.numpy()
        result1 = np.hstack([skull[0,:,:,40,0], skull[0,:,:,40,1], np.logical_or(skull[0,:,:,40,0], skull[0,:,:,40,1])])
        result2 = np.hstack([np.rot90(skull[0,:,40,:,0]), np.rot90(skull[0,:,40,:,1]), np.rot90(np.logical_or(skull[0,:,40,:,0], skull[0,:,40,:,1]))])
        result3 = np.hstack([np.rot90(skull[0,40,:,:,0]), np.rot90(skull[0,40,:,:,1]), np.rot90(np.logical_or(skull[0,40,:,:,0], skull[0,40,:,:,1]))])
        result = np.vstack([result1, result2, result3])
        result_stacked = np.stack([result,result,result], axis=-1)

        image = keras.preprocessing.image.array_to_img(result_stacked)
        image.save(self.save_path + "generated_skull_on_epoch_{epoch}.png".format(epoch=epoch))

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



if __name__ == "__main__":
    disc = make_DCGAN3D_discriminator()
    keras.utils.plot_model(disc, "discriminator.png", show_shapes=True)
    gen = make_DCGAN3D_generator()
    keras.utils.plot_model(gen, "generator.png", show_shapes=True)