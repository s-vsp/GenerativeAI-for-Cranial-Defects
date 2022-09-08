import numpy as np
import tensorflow as tf

from utils import create_dataset, latent_space_visualization

def run():
    data = create_dataset(dataset_type=0)
    data = list(data.as_numpy_iterator())

    visualization = latent_space_visualization(data)

if __name__ == "__main__":
    run()