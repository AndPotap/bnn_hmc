import os
import sys
from jax.config import config
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as onp
import time
from jax.experimental.callback import rewrite
import tqdm
import copy
import optax
import functools
from matplotlib import pyplot as plt
from bnn_hmc.core import sgmcmc
from bnn_hmc.utils import data_utils
from bnn_hmc.utils import models
from bnn_hmc.utils import losses
from bnn_hmc.utils import checkpoint_utils
from bnn_hmc.utils import cmd_args_utils
from bnn_hmc.utils import logging_utils
from bnn_hmc.utils import train_utils
from bnn_hmc.utils import tree_utils
from bnn_hmc.utils import precision_utils
from bnn_hmc.utils import optim_utils

from bnn_hmc.experiments.experiment_fns import inv_softplus
from bnn_hmc.experiments.experiment_fns import make_model
from bnn_hmc.experiments.experiment_fns import resample_params

noise_std = 0.02
invsp_noise_std = inv_softplus(noise_std)
data_npz = onp.load("./notebooks/synth_reg_data.npz")
x_ = jnp.asarray(data_npz["x_"])
y_ = jnp.asarray(data_npz["y_"])
f_ = jnp.asarray(data_npz["f_"])
x = jnp.asarray(data_npz["x"])
y = jnp.asarray(data_npz["y"])
f = jnp.asarray(data_npz["f"])
data_info = {"y_scale": 1.}

num_devices = len(jax.devices())
train_set = (f, y)
test_set = (f_, y_)
# The code below increments the shape of the data by the number of devices
train_set = data_utils.pmap_dataset(train_set, num_devices)
test_set = data_utils.pmap_dataset(test_set, num_devices)

net_fn = make_model(layer_dims=[100, 100, 100], invsp_noise_std=invsp_noise_std)
net = hk.transform_with_state(net_fn)
net_apply, net_init = net.apply, net.init
net_apply = precision_utils.rewrite_high_precision(net_apply)

prior_std = 0.1
weight_decay = 1 / prior_std**2

task = data_utils.Task("regression")
(likelihood_factory, predict_fn, ensemble_upd_fn, _, _) = train_utils.get_task_specific_fns(task, data_info)
log_prior_fn, log_prior_diff_fn = losses.make_gaussian_log_prior(weight_decay, 1.)

log_prior_fn, log_prior_diff_fn = (
        losses.make_gaussian_log_prior(weight_decay, 1.))
log_likelihood_fn = losses.make_gaussian_likelihood(1.)

lr_schedule = optim_utils.make_constant_lr_schedule_with_cosine_burnin(3.e-8, 3.e-8, 1)

param_seed = 0
params, net_state = net.init(jax.random.PRNGKey(param_seed), (f, None), True)
params = resample_params(param_seed, params, std=0.05)
optimizer = sgmcmc.sgld_gradient_update(lr_schedule, 0)

sgmcmc_train_epoch = train_utils.make_sgd_train_epoch(
    net_apply, log_likelihood_fn, log_prior_fn, optimizer, num_batches=1)

# num_iterations = int(1.e5)
# save_freq = int(1.e3)
num_iterations = 100
save_freq = int(1.e1)
all_test_preds = []
key = jax.random.PRNGKey(0)
key = jax.random.split(key, num_devices)
opt_state = optimizer.init(params)

for iteration in tqdm.tqdm(range(num_iterations)):
    params, net_state, opt_state, logprob_avg, key = sgmcmc_train_epoch(
        params, net_state, opt_state, train_set, key)
    if iteration % save_freq == 0:
        test_predictions = onp.asarray(
              predict_fn(net_apply, params, net_state, test_set))
        all_test_preds.append(test_predictions)


for pred in all_test_preds:
    # plt.plot(x_, pred[0, :, 0])
    # plt.plot(x, pred.take(1)[0, :, 0])
    plt.plot(x, pred[1][0, :, 0])
 
plt.plot(x, y, "ro")
plt.show()
