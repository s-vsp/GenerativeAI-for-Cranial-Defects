import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, models, metrics


def make_vnet(input_shape: int=128):
    inputs = layers.Input(shape=(input_shape,input_shape,input_shape,1), name="Input_layer")

    # Block 1
    conv1 = layers.Conv3D(8, (5,5,5), (1,1,1), padding="same")(inputs)
    bn1 = layers.BatchNormalization()(conv1)
    prelu1 = layers.PReLU()(bn1)
    conv2 = layers.Conv3D(8, (5,5,5), (1,1,1), padding="same")(prelu1)
    bn2 = layers.BatchNormalization()(conv2)
    prelu2 = layers.PReLU()(bn2)
    add1 = layers.Add()([prelu1, prelu2])
    down_conv1 = layers.Conv3D(16, (2,2,2), (2,2,2))(add1)
    bn3 = layers.BatchNormalization()(down_conv1)
    prelu3 = layers.PReLU()(bn3)

    # Block 2
    conv3 = layers.Conv3D(16, (5,5,5), (1,1,1), padding="same")(prelu3)
    bn4 = layers.BatchNormalization()(conv3)
    prelu4 = layers.PReLU()(bn4)
    conv4 = layers.Conv3D(16, (5,5,5), (1,1,1), padding="same")(prelu4)
    bn5 = layers.BatchNormalization()(conv4)
    prelu5 = layers.PReLU()(bn5)
    add2 = layers.Add()([prelu3, prelu5])
    down_conv2 = layers.Conv3D(32, (2,2,2), (2,2,2))(add2)
    bn6 = layers.BatchNormalization()(down_conv2)
    prelu6 = layers.PReLU()(bn6)

    # Block 3
    conv5 = layers.Conv3D(32, (5,5,5), (1,1,1), padding="same")(prelu6)
    bn7 = layers.BatchNormalization()(conv5)
    prelu7 = layers.PReLU()(bn7)
    conv6 = layers.Conv3D(32, (5,5,5), (1,1,1), padding="same")(prelu7)
    bn8 = layers.BatchNormalization()(conv6)
    prelu8 = layers.PReLU()(bn8)
    conv7 = layers.Conv3D(32, (5,5,5), (1,1,1), padding="same")(prelu8)
    bn9 = layers.BatchNormalization()(conv7)
    prelu9 = layers.PReLU()(bn9)
    add3 = layers.Add()([prelu6, prelu9])
    down_conv3 = layers.Conv3D(64, (2,2,2), (2,2,2))(add3)
    bn10 = layers.BatchNormalization()(down_conv3)
    prelu10 = layers.PReLU()(bn10)

    # Block 4
    conv8 = layers.Conv3D(64, (5,5,5), (1,1,1), padding="same")(prelu10)
    bn11 = layers.BatchNormalization()(conv8)
    prelu11 = layers.PReLU()(bn11)
    conv9 = layers.Conv3D(64, (5,5,5), (1,1,1), padding="same")(prelu11)
    bn12 = layers.BatchNormalization()(conv9)
    prelu12 = layers.PReLU()(bn12)
    conv10 = layers.Conv3D(64, (5,5,5), (1,1,1), padding="same")(prelu12)
    bn13 = layers.BatchNormalization()(conv10)
    prelu13 = layers.PReLU()(bn13)
    add4 = layers.Add()([prelu10, prelu13])
    down_conv4 = layers.Conv3D(128, (2,2,2), (2,2,2))(add4)
    bn14 = layers.BatchNormalization()(down_conv4)
    prelu14 = layers.PReLU()(bn14)

    # Block 5
    conv11 = layers.Conv3D(128, (5,5,5), (1,1,1), padding="same")(prelu14)
    bn15 = layers.BatchNormalization()(conv11)
    prelu15 = layers.PReLU()(bn15)
    conv12 = layers.Conv3D(128, (5,5,5), (1,1,1), padding="same")(prelu15)
    bn16 = layers.BatchNormalization()(conv12)
    prelu16 = layers.PReLU()(bn16)
    conv13 = layers.Conv3D(128, (5,5,5), (1,1,1), padding="same")(prelu16)
    bn17 = layers.BatchNormalization()(conv13)
    prelu17 = layers.PReLU()(bn17)
    add5 = layers.Add()([prelu14, prelu17])
    up_conv1 = layers.Conv3DTranspose(128, (2,2,2), (2,2,2))(add5)
    bn18 = layers.BatchNormalization()(up_conv1)
    prelu18 = layers.PReLU()(bn18)

    # Block 6
    concat1 = layers.Concatenate()([add4, prelu18])
    conv14 = layers.Conv3D(128, (5,5,5), (1,1,1), padding="same")(concat1)
    bn19 = layers.BatchNormalization()(conv14)
    prelu19 = layers.PReLU()(bn19)
    conv15 = layers.Conv3D(128, (5,5,5), (1,1,1), padding="same")(prelu19)
    bn20 = layers.BatchNormalization()(conv15)
    prelu20 = layers.PReLU()(bn20)
    conv16 = layers.Conv3D(128, (5,5,5), (1,1,1), padding="same")(prelu20)
    bn21 = layers.BatchNormalization()(conv16)
    prelu21 = layers.PReLU()(bn21)
    add6 = layers.Add()([prelu18, prelu21])
    up_conv2 = layers.Conv3DTranspose(64, (2,2,2), (2,2,2))(add6)
    bn22 = layers.BatchNormalization()(up_conv2)
    prelu22 = layers.PReLU()(bn22)

    # Block 7
    concat2 = layers.Concatenate()([add3, prelu22])
    conv17 = layers.Conv3D(64, (5,5,5), (1,1,1), padding="same")(concat2)
    bn23 = layers.BatchNormalization()(conv17)
    prelu23 = layers.PReLU()(bn23)
    conv18 = layers.Conv3D(64, (5,5,5), (1,1,1), padding="same")(prelu23)
    bn24 = layers.BatchNormalization()(conv18)
    prelu24 = layers.PReLU()(bn24)
    conv19 = layers.Conv3D(64, (5,5,5), (1,1,1), padding="same")(prelu24)
    bn25 = layers.BatchNormalization()(conv19)
    prelu25 = layers.PReLU()(bn25)
    add7 = layers.Add()([prelu22, prelu25])
    up_conv3 = layers.Conv3DTranspose(32, (2,2,2), (2,2,2))(add7)
    bn26 = layers.BatchNormalization()(up_conv3)
    prelu26 = layers.PReLU()(bn26)

    # Block 8
    concat3 = layers.Concatenate()([add2, prelu26])
    conv20 = layers.Conv3D(32, (5,5,5), (1,1,1), padding="same")(concat3)
    bn27 = layers.BatchNormalization()(conv20)
    prelu27 = layers.PReLU()(bn27)
    conv21 = layers.Conv3D(32, (5,5,5), (1,1,1), padding="same")(prelu27)
    bn28 = layers.BatchNormalization()(conv21)
    prelu28 = layers.PReLU()(bn28)
    add8 = layers.Add()([prelu26, prelu28])
    up_conv4 = layers.Conv3DTranspose(16, (2,2,2), (2,2,2))(add8)
    bn29 = layers.BatchNormalization()(up_conv4)
    prelu29 = layers.PReLU()(bn29)

    # Block 9
    concat4 = layers.Concatenate()([add1, prelu29])
    conv22 = layers.Conv3D(16, (5,5,5), (1,1,1), padding="same")(concat4)
    bn30 = layers.BatchNormalization()(conv22)
    prelu30 = layers.PReLU()(bn30)
    add9 = layers.Add()([prelu29, prelu30])
    outputs = layers.Conv3D(1, (1,1,1), (1,1,1), padding="same", activation="sigmoid")(add9)
    
    v_net_model = models.Model(inputs=inputs, outputs=outputs)
    return v_net_model


def dice_loss(predicted_mask, real_mask):
    eps = 1
    return 1 - (2 * np.sum(predicted_mask * real_mask) / (np.sum(predicted_mask**2 + real_mask**2) + eps))


class VNet_model(models.Model):
    def __init__(self, vnet):
        super(VNet_model, self).__init__()
        self.vnet = vnet

    def compile(self, optimizer, loss_fn):
        super(VNet_model, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_metric = metrics.Mean(name="dice_loss")
    
    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, data):
        defected_skull = data[:,:,:,:,0]
        implant = data[:,:,:,:,1]

        with tf.GradientTape() as tape:
            mask = self.vnet(defected_skull)
            loss = self.loss_fn(mask, implant)
        grads = tape.gradient(loss, self.vnet.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vnet.trainable_variables))

        self.loss_metric.update_state(loss)
        return {"dice_loss": self.loss_metric.result()}


class VNet_monitor(keras.callbacks.Callback):
    def __init__(self, save_path, skull):
        self.save_path = save_path
        self.defected_skull = skull[0,:,:,:,0]
        self.implant = skull[0,:,:,:,1]

    def on_epoch_end(self, epoch, logs=None):
        mask = self.model.vnet.predict(self.defected_skull)

        input_implant = self.skull.numpy()
        
        proj_1 = np.hstack([input_implant[0,:,:,40], mask[0,:,:,40]])
        proj_2 = np.hstack([np.rot90(input_implant[0,:,40,:]), np.rot90(mask[0,:,40,:])])
        proj_3 = np.hstack([np.rot90(input_implant[0,40,:,:]), np.rot90(mask[0,40,:,:])])
        result = np.vstack([proj_1, proj_2, proj_3])

        # For RGB-like purposes of keras arr-to-img visualization
        result_stacked = np.stack([result, result, result], axis=-1)

        image_channels = keras.preprocessing.image.array_to_img(result_stacked)
        image_channels.save(self.save_path + "implant_and_segmentation_on_{epoch}.png".format(epoch=epoch))


if __name__ == "__main__":
    vnet = make_vnet(128)
    keras.utils.plot_model(vnet, "VNET.png", show_shapes=True)