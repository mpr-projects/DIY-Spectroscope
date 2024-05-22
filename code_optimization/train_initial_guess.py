# This is a simple script for training three multilayer-perceptrons whose
# input is a wavelength and whose output is a spectral response function with
# three components (red, green and blue). The input is normalized to range
# [0, 1], based on the nm_min and nm_max params given in the settings file.
import os
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from typing import Sequence
import flax.linen as nn
import optax


# read settings file
with open('settings.yaml', 'r') as f:
    settings = yaml.safe_load(f)

nm_min = settings['nm_range']['nm_min']
nm_max = settings['nm_range']['nm_max']
model_size = settings['model_size']

settings = settings['initial_training']

fpath_crf = settings['paths']['combined_response_function']
save_path = settings['paths']['save_path']

key = jax.random.key(settings['training']['seed'])
batch_size = settings['training']['batch_size']
n_epochs = settings['training']['n_epochs']
schedule_params = settings['training']['linear_lr_schedule_params']


# load data
crf = np.load(fpath_crf)
crf_nms, crf_data = crf[0], crf[1:]
mask = np.logical_not(np.isclose(crf_data.sum(axis=0), 0))
crf_nms = crf_nms[mask][:, None]
crf_nms = (crf_nms - nm_min) / (nm_max - nm_min)
mask = np.stack((mask,)*3)
crf_data = crf_data[mask].reshape(3, -1).T


# visualization helper
def visualize(params, x, title=None):
    preds = np.asarray(apply(params, x))
    colors = ['red', 'green', 'blue']

    for i in range(3):
        plt.plot(np.asarray(x), np.asarray(preds[:, i]), c=colors[i])
        plt.plot(np.asarray(x), crf_data[:, i], c=colors[i], linestyle='--')

    # plt.legend()
    plt.title(title)
    plt.show()


# model
class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.tanh(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    x = nn.sigmoid(x)
    return x


class ResponseFunctions(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        features = self.features + (1,)
        r = MLP(features)(x)
        g = MLP(features)(x)
        b = MLP(features)(x)
        return jnp.concatenate((r, g, b), axis=-1)


model = ResponseFunctions(model_size)
apply = jax.vmap(model.apply, in_axes=(None, 0))

key, subkey = jax.random.split(key)
params = model.init(subkey, crf_nms[0])


# optimizer and loss function
lr_schedule = optax.linear_schedule(**schedule_params)
optimizer = optax.adam(learning_rate=lr_schedule)
opt_state = optimizer.init(params)


def loss_fn(params, x, y):
    preds = apply(params, x)
    return optax.l2_loss(preds, y).mean()


grad_fn = jax.jit(jax.grad(loss_fn))

visualize(params, crf_nms, title='pre')


for i in range(n_epochs):
    print(f'epoch {i}', end='             \r')
    key, subkey = jax.random.split(key)
    inds = jax.random.randint(subkey, (batch_size,), 0, len(crf_nms))
    x, y = crf_nms[inds], crf_data[inds]

    grads = grad_fn(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    if False and i % 1000 == 0:  # currently disabled
        visualize(params, crf_nms, title=f'epoch {i}')


visualize(params, crf_nms)


# save parameters
with open(save_path, 'wb') as file:
    pickle.dump(params, file)
