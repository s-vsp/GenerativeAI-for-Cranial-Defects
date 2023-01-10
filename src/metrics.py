import os
import re
import cc3d
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import optimizers
from sklearn.manifold import TSNE
from tensorflow.python.summary.summary_iterator import summary_iterator

from neural_nets.AE3D import make_AE3D_encoder, make_AE3D_decoder, AE3D_model, AE3D_monitor, reconstruction_loss
from neural_nets.VNet import make_vnet, dice_loss, VNet_model
from utils import threshold, save_to_nrrd, load_nrrd


def dice(input1: np.ndarray, input2: np.ndarray) -> float:
    """
    Computes Dice score coefficent (DSC)
    
    Args:
        - input1: Image as np.ndarray format (in evaluation tasks it is a ground truth image)
        - input2: Image as np.ndarray format (in evaluation tasks it is a 'prediction'/generated image)
    
    Returns:
        - dc: Dice score coefficent
    """
    input1 = np.atleast_1d(input1.astype(np.bool))
    input2 = np.atleast_1d(input2.astype(np.bool))
    
    intersection = np.count_nonzero(input1 & input2)
    
    size_i1 = np.count_nonzero(input1)
    size_i2 = np.count_nonzero(input2)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc


def plot_learning_curves(data_path: pathlib.Path, model_name: str):
    """
    Plot learning curves of the training process.
    Note - I feel like this function's construction is super terrible, but didn't have time to think more :P

    Args:
        - data_path: general path to the directory where all the data is kept
        - model_name: name of the model, available ones: DCGAN, WGANGP, VAE, VAE_WGANGP, IntroVAE
    """
    logs_path = str(data_path) + "/" + model_name + "3D_data/new_logs"

    # Different losses are being monitored for different models
    if model_name == "DCGAN":
        discriminator_losses = []
        generator_losses = []

        for _, _, tfevents in os.walk(logs_path):
            for tfevent in tfevents:
                for e in summary_iterator(os.path.realpath(logs_path + "/" + tfevent)):
                    for v in e.summary.value:
                        if v.tag == "epoch_Discriminator_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            discriminator_losses.append(loss_val)
                        elif v.tag == "epoch_Generator_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            generator_losses.append(loss_val)
        
        print(discriminator_losses)
        plt.figure(figsize=(14,8), dpi=100)
        #plt.plot(generator_losses, label="Generator", color=(0.482, 0.631, 0.761), linewidth=2)
        plt.plot(discriminator_losses, label="Discriminator", color=(0.369, 0.263, 0.647), linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.title("DCGAN 3D")
        plt.grid(True)
        plt.savefig(str(data_path) + "/" + model_name + "3D_data/DCGAN3D_losses.png")

    elif model_name == "WGANGP":
        critic_losses = []
        generator_losses = []

        for _, _, tfevents in os.walk(logs_path):
            for tfevent in tfevents:
                for e in summary_iterator(os.path.realpath(logs_path + "/" + tfevent)):
                    for v in e.summary.value:
                        if v.tag == "epoch_critic_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            critic_losses.append(loss_val)
                        elif v.tag == "epoch_generator_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            generator_losses.append(loss_val)
        
        plt.figure(figsize=(14,8), dpi=100)
        plt.plot(generator_losses, label="Generator", color=(0.482, 0.631, 0.761), linewidth=2)
        plt.plot(critic_losses, label="Critic", color=(0.369, 0.263, 0.647), linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Wasserstein loss")
        plt.legend(loc="best")
        plt.title("WGAN-GP 3D")
        plt.grid(True)
        plt.savefig(str(data_path) + "/" + model_name + "3D_data/WGANGP3D_losses.png")

    elif model_name == "VAE_WGANGP":
        critic_losses = []
        generator_losses = []

        for _, _, tfevents in os.walk(logs_path):
            for tfevent in tfevents:
                for e in summary_iterator(os.path.realpath(logs_path + "/" + tfevent)):
                    for v in e.summary.value:
                        if v.tag == "epoch_critic_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            critic_losses.append(loss_val)
                        elif v.tag == "epoch_generator_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            generator_losses.append(loss_val)
        
        plt.figure(figsize=(14,8), dpi=100)
        plt.plot(generator_losses, label="Generator", color=(0.482, 0.631, 0.761), linewidth=2)
        plt.plot(critic_losses, label="Critic", color=(0.369, 0.263, 0.647), linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Wasserstein loss")
        plt.legend(loc="best")
        plt.title("VAE_WGAN-GP 3D")
        plt.grid(True)
        plt.savefig(str(data_path) + "/" + model_name + "3D_data/VAE_WGANGP3D_losses.png")
    
    elif model_name == "VAE":
        kl_losses = []
        reconstruction_losses = []
        total_losses = []

        for _, _, tfevents in os.walk(logs_path):
            for tfevent in tfevents:
                for e in summary_iterator(os.path.realpath(logs_path + "/" + tfevent)):
                    for v in e.summary.value:
                        if v.tag == "epoch_kl_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            kl_losses.append(loss_val)
                        elif v.tag == "epoch_reconstruction_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            reconstruction_losses.append(loss_val)
                        elif v.tag == "epoch_total_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            total_losses.append(loss_val)

        plt.figure(figsize=(14,8), dpi=100)
        plt.subplot(3,1,1)
        plt.plot(kl_losses, label="KL loss", color=(0.184, 0.078, 0.212), linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.grid(True)
        plt.subplot(3,1,2)
        plt.plot(reconstruction_losses, label="Reconstruction loss", color=(0.369, 0.263, 0.647), linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.grid(True)
        plt.subplot(3,1,3)
        plt.plot(total_losses, label="Total loss", color=(0.776, 0.541, 0.427), linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.title("VAE 3D")
        plt.grid(True)
        plt.savefig(str(data_path) + "/" + model_name + "3D_data/VAE3D_losses.png")

    elif model_name == "IntroVAE":
        generator_reconstruction_losses = []
        generator_total_losses = []
        inference_model_reconstruction_losses = []
        inference_model_kl_losses = []
        inference_model_total_losses = []

        for _, _, tfevents in os.walk(logs_path):
            for tfevent in tfevents:
                for e in summary_iterator(os.path.realpath(logs_path + "/" + tfevent)):
                    for v in e.summary.value:
                        if v.tag == "epoch_generator_reconstruction_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            generator_reconstruction_losses.append(loss_val)
                        elif v.tag == "epoch_generator_total_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            generator_total_losses.append(loss_val)
                        elif v.tag == "epoch_inference_model_reconstruction_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            inference_model_reconstruction_losses.append(loss_val)
                        elif v.tag == "epoch_inference_model_kl_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            inference_model_kl_losses.append(loss_val)
                        elif v.tag == "epoch_inference_model_total_loss":
                            loss_val = tf.make_ndarray(v.tensor)
                            inference_model_total_losses.append(loss_val)

        plt.figure(figsize=(14,8), dpi=100)
        plt.plot(generator_reconstruction_losses, label="Generator reconstruction loss", color=(0.184, 0.078, 0.212), linewidth=2)
        plt.plot(generator_total_losses, label="Generator total loss", color=(0.369, 0.263, 0.647), linewidth=2)
        plt.plot(inference_model_reconstruction_losses, label="Inference model reconstruction loss", color=(0.482, 0.631, 0.761), linewidth=2)
        plt.plot(inference_model_kl_losses, label="Inference model KL loss", color=(0.557, 0.172, 0.314), linewidth=2)
        plt.plot(inference_model_total_losses, label="Inference model total loss", color=(0.776, 0.541, 0.427), linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.title("IntroVAE 3D")
        plt.grid(True)
        plt.savefig(str(data_path) + "/" + model_name + "3D_data/IntroVAE3D_losses.png")
        
        # inference_model_reconstruction_losses and generator_reconstruction_losses are very very very similar and so it is
        # hard to see them on the plot
        

def latent_space_visualization(data, dim_reducer: str="encoder", image_shape: int=128, seed: int=42) -> plt.figure:
    """
    Performs t-SNE projection of the skulls in the latent space. Firstly it downstreams data to the lower dimension
    by running either PCA or Encoder-based model and then in this space t-SNE is used. 

    As skulls data is really high dimensional the Encoder approach is suggested.

    Args:
        - data: dataset of the tf.Dataset type used for visualization
        - dim_reducer: string, either 'encoder' or 'pca', the technique used for dimensionality reduction
        - image_shape: shape of the input images: input_shape x input_shape x input_shape x 2
        - seed: seed used for reproductibility
    
    Returns:
        - X_tsne: embedded representation of the data in the 2-dimensional space
    """
    if dim_reducer=="encoder":
        model = AE3D_model(encoder=make_AE3D_encoder, decoder=make_AE3D_decoder, latent_dim=200)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999), 
            reconstruction_loss_fn=reconstruction_loss
        )
        model.load_weights("./data/AE3D_data/checkpoints/AE3D_training-0005.ckpt").expect_partial()
        
        latent_data = list()
        data_iterator = iter(data)
        for _ in range(len(list(data.as_numpy_iterator()))):
            sample = next(data_iterator)
            latent_sample = model.encoder.predict(sample) # shape is (200,)
            latent_data.append(latent_sample)

        tsne = TSNE(n_components=2, random_state=seed)
        X_tsne = tsne.fit_transform(np.array(latent_data))
    elif dim_reducer=="pca":
        tsne = TSNE(n_components=2, init="pca", random_state=seed)
        X_tsne = tsne.fit_transform(data.reshape([-1,image_shape**3*2]))

    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()

    return X_tsne


def latent_space_interpolation(encoder=None, decoder=None, model_type: str="GAN", image_1: tf.Tensor=None, image_2: tf.Tensor=None, latent_1=None, latent_2=None, steps: int=5) -> tf.Tensor:
    """
    Performs the interpolation between two images in the latent space.

    Args:
        - image_1: tf.Tensor image, representing the starting point of the interpolation (for VAEs)
        - image_2: tf.Tensor image, representing the ending point of the interpolation (for VAEs)
        - latent_1: latent vector, representing the starting point of the interpolation (for GANs)
        - latent_2: latent vector, representing the ending point of the interpolation (for GANs)
        - encoder: encoding model (most likely encoding part of the AutoEncoder)
        - decoder: decoding model (most likely decoding part of the AutoEncoder)
        - model_type: string either VAE or GAN
        - steps: number of steps/changes in the latent space to perform the interpolation

    Returns:
        - reconstructions_tensor: tf.Tensor of shape: steps x image_(1/2).shape, representing reconstructions of 
    """
    if model_type == "VAE":
        latent_1 = encoder(image_1)
        latent_2 = encoder(image_2)

    alphas = np.linspace(0,1,steps)
    reconstructions = list()
    for i, alpha in enumerate(alphas):
        common_latent_vector = (1 - alpha) * latent_1 + alpha * latent_2
        common_latent_vector = tf.expand_dims(common_latent_vector, axis=0)
        reconstruction = decoder.predict(common_latent_vector)
        reconstruction_thresholded = threshold(np.squeeze(reconstruction))
        
        defected_skull = reconstruction_thresholded[:,:,:,0]
        implant = reconstruction_thresholded[:,:,:,1]

        # Postprocessing - 1. LOGICAL OPERATIONS
        implant = np.logical_xor(implant, np.logical_and(implant, defected_skull))

        # Postprocessing - 2. Connected Components Analysis
        defected_skull = cc3d.connected_components(defected_skull, connectivity=6)
        implant  = cc3d.connected_components(implant, connectivity=6)

        defected_skull = cc3d.dust(defected_skull, threshold=100, connectivity=6, in_place=False)
        defected_skull[defected_skull != 0] = 1

        implant = cc3d.dust(implant, threshold=100, connectivity=6, in_place=False)
        implant[implant != 0] = 1

        save_to_nrrd(f"data/interpolations/interpolation_defected_skull_step_{i}.nrrd".format(i), defected_skull, (0.1, 0.1, 0.1))
        save_to_nrrd(f"data/interpolations/interpolation_implant_step_{i}.nrrd".format(i), implant, (0.1, 0.1, 0.1))
        reconstructions.append(reconstruction)

    reconstructions_tensor = tf.convert_to_tensor(reconstructions)
    return reconstructions_tensor


def load_pretrained_vnet(model_name: str="WGANGP", data_amount: int=5000):
    """
    Loads a pretrained V-Net model, which was trained on a dataset specified by a model_name argument.
    
    Args:
        - model_name: name of a generative model which was used to generate synthetic data
        - data_amount: amount of data in the dataset
    
    Return:
        - model: pretrained V-Net model
    """
    IMAGE_SHAPE = 128
    LR = 1e-3
    BETA_1 = 0.9
    BETA_2 = 0.999

    vnet = make_vnet(input_shape=IMAGE_SHAPE)
    model = VNet_model(vnet=vnet)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2),
        loss_fn = dice_loss
    )
    model.load_weights("data/VNet_data/" + model_name + "/" + str(data_amount) + "/checkpoints/VNet_" + model_name + "_" + str(data_amount) + ".ckpt").expect_partial()

    return model


def predict_implants(model_name: str="WGANGP", data_amount: int=5000, model=None):
    """
    'Predicts' a compatible implant to a defect. In fact it is a segmentation task performed with the use of V-Net.

    Args:
        - model_name: name of a generative model which was used to generate synthetic data
        - data_amount: amount of data in the dataset
        - model: V-Net model
    """
    for root, dirs, files in os.walk("data/autoimpant_test/nrrd/defective_skull"):
        for file in files:
            defective_skull_path = os.path.join(root, file)
            defective_skull, spacing, _, _ = load_nrrd(defective_skull_path)
            defective_skull = tf.expand_dims(defective_skull, axis=0)
            defect_type = re.split("/", str(defective_skull_path.replace(os.sep, "/") ))[-2]
            
            implant_prediction = model.vnet.predict(defective_skull)
            save_to_nrrd("data/VNet_data/" + model_name + "/" + str(data_amount) + "/nrrd/implant/" + defect_type + "/" + file, np.squeeze(implant_prediction), spacing)


def compute_dice_for_all(model_name: str="WGANGP", data_amount: int=5000):
    """
    Computes dice score for all predictions

    Args:
        - model_name: name of a generative model which was used to generate synthetic data
        - data_amount: amount of data in the dataset

    Returns:
        - mean_dice: mean dice score for all the records in the specified dataset
    """
    dices = []
    for root, dirs, files in os.walk("data/VNet_data/" + model_name + "/" + str(data_amount) + "/nrrd/implant"):
        for file in files:
            predicted_implant_path = os.path.join(root, file)
            defect_type = re.split("/", str(predicted_implant_path.replace(os.sep, "/") ))[-2]

            predicted_implant, spacing, _, _ = load_nrrd(predicted_implant_path)
            ground_truth_implant, _, _, _ = load_nrrd("data/autoimplant_test_gt/" + defect_type + "/" + file)

            dice = dice(ground_truth_implant, predicted_implant)
            dices.append(dice)

    mean_dice = np.mean(np.array(dices))
    return mean_dice