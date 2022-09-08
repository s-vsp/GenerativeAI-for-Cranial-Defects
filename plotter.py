import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.summary.summary_iterator import summary_iterator

dirname = os.path.dirname(__file__)
logs_dir = os.path.realpath(dirname + "/data/VAE_WGANGP3D_data/logs/train")

critic_losses = []
generator_losses = []

for _, _, tfevents in os.walk(logs_dir):
    for tfevent in tfevents:
        for e in summary_iterator(os.path.realpath(logs_dir + "/" + tfevent)):
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
plt.grid(True)
plt.show()