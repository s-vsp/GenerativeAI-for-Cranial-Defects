import os
import re
import pathlib
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import io
import scipy.ndimage
import tensorflow as tf
from tensorflow import keras


def load_nrrd(path: pathlib.Path) -> tuple:
    """
    Function loading 3D nrrd file.
    Original image loaded has given directions as follows: (Z, Y, X) where Z - depth, Y - heigth, X - width.
    From more clearly and natural overlook we want them to be represented as (Y, X, Z) 

    Args:
        - path: path to the nrrd file
    
    Returns:
        - volume: (y_slice, x_slice, z_slice)
        - spacing: spacing between volumes
        - origin: image origin
        - direction: image direction
    """
    image = sitk.ReadImage(path)
    spacing = image.GetSpacing()
    volume = (sitk.GetArrayFromImage(image).swapaxes(0, 1)).swapaxes(1, 2)
    origin = image.GetOrigin()
    direction = image.GetDirection()

    if image.GetDimension() != 3:
        raise ValueError("Incorrect dimensionality of loaded nrrd file.")
    
    return volume, spacing, origin, direction


def save_to_nrrd(save_path: pathlib.Path, volume: np.ndarray, spacing: tuple, origin: tuple=None, direction: tuple=None):
    image = sitk.GetImageFromArray(volume.swapaxes(2, 1).swapaxes(1, 0).astype(np.uint8))
    image.SetSpacing(spacing)
    if origin is not None:
        image.SetOrigin(origin)
    else:
        image.SetOrigin((0.0, 0.0, 0.0))
    if direction is not None:
        image.setDirection(direction)
    sitk.WriteImage(image, save_path)


def downsample(image: np.ndarray, downsampling_factor: int=4):
    xx, yy, zz = np.meshgrid(
        np.arange(image.shape[0] // downsampling_factor), 
        np.arange(image.shape[1] // downsampling_factor),
        np.arange(image.shape[2] // downsampling_factor)
        )
    xx *= downsampling_factor
    yy *= downsampling_factor
    zz *= downsampling_factor
    downsampled_image = scipy.ndimage.map_coordinates(image, [yy, xx, zz], order=0)
    return downsampled_image


def concat_channels_0(filename) -> tf.Tensor:
    """
    Utility function that maps the first image of shape 512 x 512 x 512 to a second image of the same shape,
    but being it's other form. The mapping is done on: defective skull <-> implant.
    Then both images are downsampled to 128 x 128 x 128, their dimensionalities are expanded to 
    128 x 128 x 128 x 1 and finally they are concatenated in the last channel resulting in the tf.Tensor 
    of shape 128 x 128 x 128 x 2

    Args:
        - filename: Tensor of string dtype representing one first input image form
    
    Returns:
        - out: output 'image' tensor of shape 128 x 128 x 128 x 2
    """
    # Get the path of the input and then get the path to it's other form
    first_form = filename.numpy().decode("utf8")
    second_form = filename.numpy().decode("utf8").replace("defective_skull", "implant")

    volume_1, _, _, _ = load_nrrd(first_form)
    volume_2, _, _, _ = load_nrrd(second_form)

    volume_1_downsampled = downsample(volume_1, 4)
    volume_2_downsampled = downsample(volume_2, 4)

    channel_1 = tf.expand_dims(volume_1_downsampled, -1)
    channel_2 = tf.expand_dims(volume_2_downsampled, -1)
    
    out = tf.concat([channel_1, channel_2], -1)
    out = tf.cast(out, tf.float32)
    return out


def concat_channels_1(filename) -> tf.Tensor:
    # TODO: Check why the complete skull channel doesn't load.
    """
    Utility function that maps the first image of shape 512 x 512 x 512 to a second image of the same shape,
    but being it's other form. The mapping is done on: complete skull <-> implant.
    Then both images are downsampled to 128 x 128 x 128, their dimensionalities are expanded to 
    128 x 128 x 128 x 1 and finally they are concatenated in the last channel resulting in the tf.Tensor 
    of shape 128 x 128 x 128 x 2

    Args:
        - filename: Tensor of string dtype representing one first input image form
    
    Returns:
        - out: output 'image' tensor of shape 128 x 128 x 128 x 2
    """
    # Get the path of the input and then get the path to it's other form
    first_form = filename.numpy().decode("utf8")

    if "bilateral" in first_form:
        second_form = filename.numpy().decode("utf8").replace(r"/implant/bilateral/", r"/complete_skull/")
    elif "frontoorbital" in first_form:
        second_form = filename.numpy().decode("utf8").replace(r"/implant/frontoorbital/", r"/complete_skull/")
    elif "parietotemporal" in first_form:
        second_form = filename.numpy().decode("utf8").replace(r"/implant/parietotemporal/", r"/complete_skull/")
    elif "random_1" in first_form:
        second_form = filename.numpy().decode("utf8").replace(r"/implant/random_1/", r"/complete_skull/")
    elif "random_2" in first_form:
        second_form = filename.numpy().decode("utf8").replace(r"/implant/random_2/", r"/complete_skull/")
    
    
    volume_1, _, _, _ = load_nrrd(first_form)
    volume_2, _, _, _ = load_nrrd(second_form)
    
    volume_1_downsampled = downsample(volume_1, 4)
    volume_2_downsampled = downsample(volume_2, 4)

    channel_1 = tf.expand_dims(volume_1_downsampled, -1)
    channel_2 = tf.expand_dims(volume_2_downsampled, -1)
    
    out = tf.concat([channel_1, channel_2], -1)
    out = tf.cast(out, tf.float32)
    return out


def map_concatenate_0(filename, output_shape):
    """
    Utility function to perform mapping on tf.data.Dataset, it follows the neccessity of setting the shape
    of data in Dataset object due to possibility of anything-output resulting in <unknown> shape.

    Args:
        - filename: Tensor of string dtype representing one first input image form
        - output_shape: shape of the output data
    
    Returns:
        - out: output 'image' tensor of shape 128 x 128 x 128 x 2
    """
    out = tf.py_function(concat_channels_0, [filename], tf.float32)
    out.set_shape(output_shape)
    return out


def map_concatenate_1(filename, output_shape):
    """
    Utility function to perform mapping on tf.data.Dataset, it follows the neccessity of setting the shape
    of data in Dataset object due to possibility of anything-output resulting in <unknown> shape.

    Args:
        - filename: Tensor of string dtype representing one first input image form
        - output_shape: shape of the output data
    
    Returns:
        - out: output 'image' tensor of shape 128 x 128 x 128 x 2
    """
    out = tf.py_function(concat_channels_1, [filename], tf.float32)
    out.set_shape(output_shape)
    return out


def create_dataset(dataset_type: int=0, output_shape: tuple=(128,128,128,2), batch_size: int=16):
    """
    Function that generates tf.data.Dataset based on given conditions

    Args:
        - dataset_type: type of the dataset:
            0 - Dataset where ich data record is a pair of defected skull and implant for it.
                They are represented as tf.Tensors of the given shape (e.g. 512 x 512 x 512) and 2 channels
                resulting in the tensor of shape 512 x 512 x 512 x 2
    """
    # Load data
    complete_skulls = tf.data.Dataset.list_files("./data/autoimplant/nrrd/complete_skull/*.nrrd", shuffle=False)

    if dataset_type == 0:
        defective_skulls_bilateral = tf.data.Dataset.list_files("./data/autoimplant/nrrd/defective_skull/bilateral/*.nrrd", shuffle=False)
        defective_skulls_frontoorbital = tf.data.Dataset.list_files("./data/autoimplant/nrrd/defective_skull/frontoorbital/*.nrrd", shuffle=False)
        defective_skulls_parietotemporal = tf.data.Dataset.list_files("./data/autoimplant/nrrd/defective_skull/parietotemporal/*.nrrd", shuffle=False)
        defective_skulls_random_1 = tf.data.Dataset.list_files("./data/autoimplant/nrrd/defective_skull/random_1/*.nrrd", shuffle=False)
        defective_skulls_random_2 = tf.data.Dataset.list_files("./data/autoimplant/nrrd/defective_skull/random_2/*.nrrd", shuffle=False)

        # Concatenate all defected skulls
        defective_skulls = defective_skulls_bilateral.concatenate(defective_skulls_frontoorbital).concatenate(defective_skulls_parietotemporal).concatenate(defective_skulls_random_1).concatenate(defective_skulls_random_2)
        
        dataset = defective_skulls.map(lambda filename: map_concatenate_0(filename, output_shape), num_parallel_calls=tf.data.AUTOTUNE)
    
    elif dataset_type == 1:
        implants_bilateral = tf.data.Dataset.list_files("./data/autoimplant/nrrd/implant/bilateral/*.nrrd", shuffle=False)
        implants_frontoorbital = tf.data.Dataset.list_files("./data/autoimplant/nrrd/implant/frontoorbital/*.nrrd", shuffle=False)
        implants_parietotemporal = tf.data.Dataset.list_files("./data/autoimplant/nrrd/implant/parietotemporal/*.nrrd", shuffle=False)
        implants_random_1 = tf.data.Dataset.list_files("./data/autoimplant/nrrd/implant/random_1/*.nrrd", shuffle=False)
        implants_random_2 = tf.data.Dataset.list_files("./data/autoimplant/nrrd/implant/random_2/*.nrrd", shuffle=False)

        # Concatenate all implants
        implants = implants_bilateral.concatenate(implants_frontoorbital).concatenate(implants_parietotemporal).concatenate(implants_random_1).concatenate(implants_random_2)

        dataset = implants.map(lambda filename: map_concatenate_1(filename, output_shape), num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.shuffle(570)
    
