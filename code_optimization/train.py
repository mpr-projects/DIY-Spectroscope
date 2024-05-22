# This is a (relatively) quick an dirty script to make the response functions
# of our DIY spectroscope consistent with the signal data from lamps and
# the sky.
import os
import copy
import yaml
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

# calculation of Planck's law requires higher accuracy
jax.config.update("jax_enable_x64", True)

from typing import Sequence
import flax.linen as nn
import optax

from helper_interpolation import pw_linear
from helper_color_temp_jax import get_radiance


# read settings file
with open('settings.yaml', 'r') as f:
    settings = yaml.safe_load(f)

nm_min_ = settings['nm_range']['nm_min']  # nms of normalization
nm_max_ = settings['nm_range']['nm_max']
model_size = settings['model_size']

settings = settings['main_training']

paths = settings['paths']
fpath_model = paths['model']
fpath_weights = paths['weights']
fpath_sky = paths['sky']
fpath_ref = paths['ref']
save_path_model = paths['save_path_model']
save_path_rfs = paths['save_path_rfs']
save_path_color_temps = paths['save_path_color_temps']

training = settings['training']
batch_size = training['batch_size']
n_val_samples = training['n_val_samples']
n_epochs = training['n_epochs']
key = jax.random.key(training['seed'])
schedule_params = training['warmup_cosine_decay_lr_schedule_params']
optimizer_params = training['optimizer_params']
loss_fn_weights = training['loss_fn_weights']

debug = training['debug']


# load initial (pre-trained) params
with open(fpath_model, 'rb') as file:
    params = pickle.load(file)


# load data - weights of r-, g-, and b-response functions
weights = np.load(fpath_weights)
nms, weights_data = weights[0], weights[1:]

mask = np.logical_not(np.isclose(weights_data.sum(axis=0), 0))

nms_src = copy.deepcopy(nms)
nms_src_masked = nms_src[mask]
nms = nms[mask]

nm_min, nm_max = nms.min(), nms.max()  # nms of data, not of normalization
nm_median_idx = len(nms) // 2
weights_data = weights_data[:, mask]

weights_r = weights_data[0]
weights_g = weights_data[1]
weights_b = weights_data[2]

# allow excluding parts of the response function (if values are steep or small)
for w, c in zip([weights_r, weights_g, weights_b], ['red', 'green', 'blue']):
    for a, b in settings['exclude_nm_ranges'][c]:
        w[np.logical_and(nms > a, nms < b)] = 0

weights_sum = weights_r + weights_g + weights_b

weights_median = [
    weights_r[nm_median_idx],
    weights_g[nm_median_idx],
    weights_b[nm_median_idx]]
weights_median_sum = sum(weights_median)

mask_r = np.isclose(weights_r, 0)
mask_g = np.isclose(weights_g, 0)
mask_b = np.isclose(weights_b, 0)

mask_r_ = np.logical_not(mask_r)
mask_g_ = np.logical_not(mask_g)
mask_b_ = np.logical_not(mask_b)

mask_rg = np.logical_not(np.logical_or(mask_r, mask_g))
mask_bg = np.logical_not(np.logical_or(mask_b, mask_g))

weights_data = weights_data.T


# load data - signal from images of sky
sky_files = os.listdir(fpath_sky)
sky_files = [os.path.join(fpath_sky, f) for f in sky_files]

sky_data = list()

for file in sky_files:
    data = np.load(file)
    nms_, data = data[0, mask], data[1:, mask]

    if not np.allclose(nms, nms_):
        data = pw_linear(nms_, data, nms, y_axis=1)

    sky_data.append(data.T)

sky_data = np.stack(sky_data, axis=0)


# load data - signal and spectrum of reference images (lamps)
ref_files = os.listdir(fpath_ref)
ref_fnames = ['_'.join(f.split('_')[:-1]) for f in ref_files
              if f.endswith('_signal.npy')
              or f.endswith('_color-temperature.txt')]
ref_fnames = np.unique(ref_fnames)
ref_files = [os.path.join(fpath_ref, f) for f in ref_fnames]

ref_lamp_names, ref_lamp_cts, ref_lamp_cts_cutoffs = list(), list(), list()

with open(os.path.join(fpath_ref, 'lamps.txt')) as f:
    contents = f.read()

for line in contents.splitlines():
    line = line.split()
    ref_lamp_names.append(line[0])
    ref_lamp_cts.append(line[1])
    ref_lamp_cts_cutoffs.append(line[2:])

ref_lamp_cts = np.stack(ref_lamp_cts, axis=0).astype(float)
ref_lamp_cts_cutoffs = np.stack(ref_lamp_cts_cutoffs, axis=0).astype(float)

params['params']['color_temps'] = jnp.array(ref_lamp_cts)

ref_signals, ref_lamps = list(), list()

for file in ref_files:
    signal = np.load(file + '_signal.npy')
    nms_, signal = signal[0, mask], signal[1:, mask]
    assert np.allclose(nms, nms_), 'Wavelengths don\'t match.'
    ref_signals.append(signal.T)

    with open(file + '_color-temperature.txt') as f:
        name = f.read().strip()
        idx = ref_lamp_names.index(name)
        ref_lamps.append(idx)


ref_signals = np.stack(ref_signals, axis=0)
ref_lamps = np.stack(ref_lamps, axis=0)

thres_l = ref_lamp_cts_cutoffs[ref_lamps][:, :1]
thres_u = ref_lamp_cts_cutoffs[ref_lamps][:, 1:]

ref_mask = np.logical_or(  # limit spectrum to range given in source file
    nms[None, :] < thres_l, nms[None, :] > thres_u)

sky_data = np.concatenate((sky_data, ref_signals), axis=0)

key, subkey = jax.random.split(key)
sky_data = jax.random.permutation(subkey, sky_data)
sky_data_val, sky_data = sky_data[:n_val_samples], sky_data[n_val_samples:]

nms_src_masked = copy.deepcopy(nms)
nms = nms[:, None]
nms = (nms - nm_min_) / (nm_max_ - nm_min_)


# visualization helper
def visualize_response(params, nms, title=None):
    preds = np.asarray(apply(params, nms))
    colors = ['red', 'green', 'blue']
    masks = [mask_r_, mask_g_, mask_b_]
    nms, preds = np.asarray(nms), np.asarray(preds)

    for i in range(3):
        m, c = masks[i], colors[i]
        plt.plot(nms_src_masked[m], preds[:, i][m], c=c, label=c)

    plt.legend()
    plt.title(title)
    plt.show()


def visualize_ref(params, nms, title=None):
    rfs = apply(params, nms)
    spectrum = ref_signals / rfs[None, :, :]

    spectrum_r = spectrum[:, :, 0]
    spectrum_g = spectrum[:, :, 1]
    spectrum_b = spectrum[:, :, 2]

    spectrum_r = spectrum_r.at[:, mask_r].set(0)
    spectrum_g = spectrum_g.at[:, mask_g].set(0)
    spectrum_b = spectrum_b.at[:, mask_b].set(0)

    spectrum = (
        weights_r*spectrum_r
        + weights_g*spectrum_g
        + weights_b*spectrum_b) / weights_sum
    spectrum = jnp.where(ref_mask, 0, spectrum)

    norm_vals = spectrum[:, nm_median_idx][:, None]
    spectrum = spectrum / norm_vals
    spectrum_r = spectrum_r / norm_vals
    spectrum_g = spectrum_g / norm_vals
    spectrum_b = spectrum_b / norm_vals

    nms_ = nms_src_masked
    cts = params['params']['color_temps']
    ref_spectra = get_radiance(cts, nms_, normalize=False)
    norm_vals = ref_spectra[:, nm_median_idx][:, None]
    ref_spectra = ref_spectra / norm_vals
    ref_spectra = jnp.where(ref_mask, 0, ref_spectra[ref_lamps])

    for idx in range(len(ref_signals)):
        plt.plot(nms_[mask_r_], spectrum_r[idx, mask_r_], c='r')
        plt.plot(nms_[mask_g_], spectrum_g[idx, mask_g_], c='g')
        plt.plot(nms_[mask_b_], spectrum_b[idx, mask_b_], c='b')
        # plt.plot(nms, spectrum[idx], c='black', alpha=0.5, label='Combined')
        idx_ = ref_lamps[idx]
        plt.plot(nms_, ref_spectra[idx_], c='brown',
                 label=f'Target {cts[idx_]:.2f}K')
        plt.title(title)
        plt.legend()
        plt.show()


def visualize_sky(params, nms, signals, title=None):
    rfs = apply(params, nms)
    spectrum = signals / rfs[None, :, :]
    spectrum_r = spectrum[:, :, 0]
    spectrum_g = spectrum[:, :, 1]
    spectrum_b = spectrum[:, :, 2]

    nms_ = nms_src_masked
    n = len(signals)
    t = None if title is None else f'{title} '

    for idx in range(n):
        plt.plot(nms_[mask_r_], spectrum_r[idx, mask_r_], c='r')
        plt.plot(nms_[mask_g_], spectrum_g[idx, mask_g_], c='g')
        plt.plot(nms_[mask_b_], spectrum_b[idx, mask_b_], c='b')
        title = ('' if t is None else t) + f'{idx+1}/{n}'
        plt.title(title)
        plt.show()


def format_timedelta(d):
    h, s = divmod(d.seconds, 3600)
    m, s = divmod(s, 60)

    res = ''

    if h != 0:
        res += f'{h}h '

    if m != 0:
        res += f'{m}m '

    res += f'{s}s'
    return res


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

apply_c = [lambda p, x: model.apply(p, x)[i] for i in range(3)]
slope_fn_c = [jax.grad(a, argnums=1) for a in apply_c]
slope_fn_c = [jax.vmap(s, in_axes=(None, 0)) for s in slope_fn_c]
slope_fn_r, slope_fn_g, slope_fn_b = slope_fn_c

lr_schedule = optax.warmup_cosine_decay_schedule(
    decay_steps=n_epochs, **schedule_params)
optimizer = optax.adamw(learning_rate=lr_schedule, **optimizer_params)
opt_state = optimizer.init(params)


def loss_fn_1(params, nms, signals, rfs=None, debug=False):
    # make all three channels produce the same spectrum
    if rfs is None:
        rfs = apply(params, nms)

    spectrum = signals / rfs[None, :, :]
    spectrum_r = spectrum[:, :, 0]
    spectrum_g = spectrum[:, :, 1]
    spectrum_b = spectrum[:, :, 2]

    w, w_sum = weights_median, weights_median_sum
    norm_vals = (
        w[0]*spectrum_r[:, nm_median_idx]
        + w[1]*spectrum_g[:, nm_median_idx]
        + w[2]*spectrum_b[:, nm_median_idx])[:, None] / w_sum

    spectrum_r = spectrum_r / norm_vals
    spectrum_g = spectrum_g / norm_vals
    spectrum_b = spectrum_b / norm_vals

    spectrum_r = spectrum_r[:, mask_rg]
    spectrum_rg = spectrum_g[:, mask_rg]
    loss_r = optax.l2_loss(spectrum_r, spectrum_rg)

    if debug is True:
        plt.plot(nms[mask_rg], spectrum_r[0], c='r')
        plt.plot(nms[mask_rg], spectrum_rg[0], c='g')
        plt.title('Loss_1, Red-Green')
        plt.show()

    spectrum_b = spectrum_b[:, mask_bg]
    spectrum_bg = spectrum_g[:, mask_bg]
    loss_b = optax.l2_loss(spectrum_b, spectrum_bg)

    if debug is True:
        plt.plot(nms[mask_bg], spectrum_b[0], c='b')
        plt.plot(nms[mask_bg], spectrum_bg[0], c='g')
        plt.title('Loss_1, Blue-Green')
        plt.show()

    loss = jnp.concatenate((loss_r, loss_b), axis=1)
    return loss.mean(), rfs


def loss_fn_2(params, rfs, debug=False):
    # make the average predicted spectra match the reference spectra
    if debug is True:
        plt.plot(nms, ref_signals[0, :, 0], c='r')
        plt.plot(nms, ref_signals[0, :, 1], c='g')
        plt.plot(nms, ref_signals[0, :, 2], c='b')
        plt.title('Loss_2, Reference 0, Signal')
        plt.show()

    spectrum = ref_signals / rfs[None, :, :]

    spectrum_r = spectrum[:, :, 0]
    spectrum_g = spectrum[:, :, 1]
    spectrum_b = spectrum[:, :, 2]

    spectrum_r = spectrum_r.at[:, mask_r].set(0)
    spectrum_g = spectrum_g.at[:, mask_g].set(0)
    spectrum_b = spectrum_b.at[:, mask_b].set(0)

    spectrum = (
        weights_r*spectrum_r
        + weights_g*spectrum_g
        + weights_b*spectrum_b) / weights_sum

    spectrum = jnp.where(ref_mask, 0, spectrum)

    norm_vals = spectrum[:, nm_median_idx][:, None]
    spectrum = spectrum / norm_vals

    cts = params['params']['color_temps']
    ref_spectra = get_radiance(
        cts, nms_src_masked, refractive_index=1.000277, normalize=False)

    norm_vals_ref = ref_spectra[:, nm_median_idx][:, None]
    ref_spectra = ref_spectra / norm_vals_ref
    ref_spectra = ref_spectra[ref_lamps]
    ref_spectra = jnp.where(ref_mask, 0, ref_spectra)

    ref_spectra = ref_spectra[ref_lamps]
    loss = optax.l2_loss(spectrum, ref_spectra).mean()

    spectrum_r = spectrum_r / norm_vals
    spectrum_r = jnp.where(ref_mask, 0, spectrum_r)
    loss_r = optax.l2_loss(spectrum_r, ref_spectra)[:, mask_r_]
    loss += loss_r.mean()

    spectrum_g = spectrum_g / norm_vals
    spectrum_g = jnp.where(ref_mask, 0, spectrum_g)
    loss_g = optax.l2_loss(spectrum_g, ref_spectra)[:, mask_g_]
    loss += loss_g.mean()

    spectrum_b = spectrum_b / norm_vals
    spectrum_b = jnp.where(ref_mask, 0, spectrum_b)
    loss_b = optax.l2_loss(spectrum_b, ref_spectra)[:, mask_b_]
    loss += loss_b.mean()

    if debug is True:
        plt.plot(nms, spectrum_r[0], c='r')
        plt.plot(nms, spectrum_g[0], c='g')
        plt.plot(nms, spectrum_b[0], c='b')
        plt.plot(nms, spectrum[0], c='black')
        plt.plot(nms, ref_spectra[0], c='brown')
        plt.title('Loss_2, Spectrum and Mask')
        plt.show()

    return loss / 4


def loss_fn_3(params, nms, debug=False):
    # prevent (near) jumps in response functions
    slope_r = slope_fn_r(params, nms)
    slope_g = slope_fn_g(params, nms)
    slope_b = slope_fn_b(params, nms)

    slope_r = jnp.abs(slope_r)
    slope_g = jnp.abs(slope_g)
    slope_b = jnp.abs(slope_b)

    loss = (slope_r.mean() + slope_g.mean() + slope_b.mean()) / 3
    return loss


def loss_fn_4(params, debug=False):
    # make color temperatures similar to those given on package
    cts = params['params']['color_temps']
    loss = optax.l2_loss(cts, ref_lamp_cts).mean()
    return loss


def loss_fn(params, nms, signals, debug=False):
    l1, rfs = loss_fn_1(params, nms, signals, debug=debug)
    l2 = loss_fn_2(params, rfs, debug=debug)
    l3 = loss_fn_3(params, nms, debug=debug)
    l4 = loss_fn_4(params, debug=debug)

    if debug is True:
        print('losses:', l1, l2, l3, l4)

    w = loss_fn_weights
    return w[0]*l1 + w[1]*l2 + w[2]*l3 + w[3]*l4


grad_fn = jax.grad(loss_fn)

if debug is False:
    grad_fn = jax.jit(grad_fn)
    slope_fn_r = jax.jit(slope_fn_r)
    slope_fn_g = jax.jit(slope_fn_g)
    slope_fn_b = jax.jit(slope_fn_b)


if debug is True:
    visualize_response(params, nms, title='Response Functions')
    visualize_ref(params, nms, title='References')
    visualize_sky(params, nms, sky_data[:4], title='Sky')
    loss_fn(params, nms, sky_data[:4], debug=True)


visualize_response(params, nms, title='pre')
visualize_ref(params, nms, title='pre')
visualize_sky(params, nms, sky_data[:2], title='pre')
# visualize_sky(params, nms, sky_data, title='pre')

best_params, best_loss, best_eid = None, float('inf'), 0
loss_fn = jax.jit(loss_fn)
start_time = datetime.datetime.now()

for i in range(n_epochs):
    print(f'epoch {i+1}/{n_epochs} {(i+1)/n_epochs*100:.2f}% ', end='')

    runtime = datetime.datetime.now() - start_time
    print(format_timedelta(runtime), end='')

    key, subkey = jax.random.split(key)
    inds = jax.random.randint(subkey, (batch_size,), 0, len(sky_data))

    grads = grad_fn(params, nms, sky_data[inds])
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    loss = loss_fn(params, nms, sky_data_val)

    if loss < best_loss:
        best_eid = i
        best_eid = i
        best_loss = loss
        best_params = copy.deepcopy(params)

    print(f', loss={loss:.8f}, best={best_loss:.8f} @ {best_eid}',
          end='             \r')

    if i % 10000 == 0:
        cts = params['params']['color_temps']
        print(f'\ncts = {cts}\n')
        continue
        visualize_response(params, nms, title=f'epoch {i}')
        visualize_ref(params, nms, title=f'epoch {i}')
        visualize_sky(params, nms, sky_data[:2], title=f'epoch {i}')

print('\nTraining Finished')

visualize_response(best_params, nms, title='post')
visualize_ref(best_params, nms, title='post')
# visualize_sky(best_params, nms, sky_data[:2], title='post')
visualize_sky(best_params, nms, sky_data, title='post')


# save parameters
with open(save_path_model, 'wb') as file:
    pickle.dump(best_params, file)


# save response functions
rfs_ = apply(best_params, nms)
rfs_ = rfs_.at[mask_r, 0].set(0)
rfs_ = rfs_.at[mask_g, 1].set(0)
rfs_ = rfs_.at[mask_b, 2].set(0)
rfs_ /= rfs_.max()
rfs = np.zeros((3, len(nms_src)))
rfs[:, mask] = rfs_.T

if debug is True:
    plt.plot(nms_src, rfs[0], c='r')
    plt.plot(nms_src, rfs[1], c='g')
    plt.plot(nms_src, rfs[2], c='b')
    plt.title('Exported Response Functions')
    plt.show()

vals = np.concatenate((nms_src[None, :], rfs), axis=0)
np.save(save_path_rfs, vals)


# save color temperatures
cts = params['params']['color_temps']
vals = ''
for n, v in zip(ref_lamp_names, cts):
    vals += f'{n}: {v}\n'

with open(save_path_color_temps, 'w') as f:
    f.write(vals)
