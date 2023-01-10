# GenerativeAI-for-Cranial-Defects
Repo for bachelor thesis.

## General structure:
There are two main folders **data** and **src**:
- **data** includes all the data used and generated in the scope of this work. Due to it's huge amounts it is not stored directly in the repository, however to obtain it, contact me.
- **src** includes neural networks, experiments and other functionalities used.
- **requirements.yml** includes required packages used for setting up the virtual environment

### The general structure
```
GenerativeAI-for-Cranial-Defects
|
|----- data
|       |----- DCGAN3D_data
|       |           |----- checkpoints
|       |           |----- images
|       |           |----- logs
|       |----- IntroVAE3D_data
|       |           |----- checkpoints
|       |           |----- images
|       |           |----- logs
|       |----- VAE3D_data
|       |           |----- checkpoints
|       |           |----- images
|       |           |----- logs
|       |----- WGANGP3D_data
|       |           |----- checkpoints
|       |           |----- images
|       |           |----- logs
|       |----- VAE_WGANGP3D_data
|       |           |----- checkpoints
|       |           |----- images
|       |           |----- logs
|       |----- AE3D_data
|                   |----- checkpoints
|                   |----- images
|----- src
|       |----- experiments
|       |           |----- run_AE3D.py
|       |           |----- run_DCGAN3D.py
|       |           |----- run_IntroVAE3D.py
|       |           |----- run_VAE3D.py
|       |           |----- run_VAE_WGANGP3D.py
|       |           |----- run_WGANGP3D.py
|       |----- neural_nets
|       |           |----- AE3D.py
|       |           |----- DCGAN3D.py
|       |           |----- IntroVAE3D.py
|       |           |----- VAE_WGANGP3D.py
|       |           |----- VAE3D.py
|       |           |----- VNet.py
|       |           |----- WGANGP3D.py
|       |----- encode_with_AE.py
|       |----- generate_skulls_WGANGP3D.py
|       |----- interpolation.py
|       |----- metrics.py
|       |----- utils.py
|----- requirements.yml
|----- .gitignore
```


## Data
In the **data** directory there are 5 other directories, namely - DCGAN3D_data, WGANGP3D_data, VAE3D_data, VAE_WGANGP3D_data and IntroVAE3D_data which include data from the training process of these models. Each of them includes 3 other subdirectories, which are checkpoints, logs and images. Checkpoints include model checkpoints from the training process of this model. Logs include metrics monitored in the training process. Images include examples of skulls generated at each epoch of the model. What's more there is also a AE3D_data directory, wchich mainly stores data used for dimensionality reduction before running t-SNE or UMAP.

## Src
In the **src** directory there are 2 other directories - experiments and neural_nets. Neural_nets simply includes each of the models code and experiments includes training scripts for these models. Other files in this directory are:
- utils.py: includes main utility functions used in the nets, preprocessing tasks and experiments
- metrics.py: metrics used for the evaluation of the models
- encode_with_AE.py: autoencoder used for first reduction of dimensionalities of the data for t-SNE and UMAP
- generate_skulls_WGANGP3D.py: used for generating skulls with WGAN-GP model used for interpolation representation # TODO: update for other models
- interpolation.py: performs interpolation between two latent vectors
