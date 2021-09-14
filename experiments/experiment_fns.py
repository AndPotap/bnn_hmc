from jax import numpy as jnp
import jax
import haiku as hk


def inv_softplus(x):
    return jnp.log(jnp.exp(x) - 1)


# Noise in the data
noise_std = 0.02
invsp_noise_std = inv_softplus(noise_std)


def make_model(layer_dims, invsp_noise_std):
    def forward(batch, is_training):
        x, _ = batch
        x = hk.Flatten()(x)
        for layer_dim in layer_dims:
            x = hk.Linear(layer_dim)(x)
            x = jax.nn.relu(x)
        x = hk.Linear(1)(x)
        x = jnp.concatenate([x, jnp.ones_like(x) * invsp_noise_std], -1)
        return x
    return forward


def resample_params(seed, params, std=0.005):
    key = jax.random.PRNGKey(seed)
    num_leaves = len(jax.tree_leaves(params))
    normal_keys = list(jax.random.split(key, num_leaves))
    treedef = jax.tree_structure(params)
    normal_keys = jax.tree_unflatten(treedef, normal_keys)
    params = jax.tree_multimap(lambda p, k: jax.random.normal(k, p.shape) * std,
                               params, normal_keys)
    return params
