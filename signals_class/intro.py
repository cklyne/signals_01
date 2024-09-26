import jax
import nifty8.re as jft
from jax import numpy as jnp
from matplotlib import pyplot as plt
from sg_class.signal_classes import (
    FixedPowerCorrelatedField,
    FixedPowerCorrelatedFieldNonLin,
    SignalResponse,
)

jax.config.update("jax_enable_x64", True)
SEED = 42


def amplitude_spectrum(grid):
    k = grid.harmonic_grid.mode_lengths
    return 0.02 / (1 + k**2)


if __name__ == "__main__":
    key = jax.random.PRNGKey(SEED)

    dims = (128, 128)
    distances = 1 / dims[0]
    grid = jft.correlated_field.make_grid(
        dims,
        distances=distances,
        harmonic_type="fourier",
    )
    a = amplitude_spectrum(grid)

    # fig = plt.plot()
    # plt.title("Raw amplitudes")
    # plt.plot(a)
    # plt.xlabel("index")
    # plt.ylabel("a")
    # plt.show()

    a = a[grid.harmonic_grid.power_distributor]
    noise_std = 0.1
    signal = FixedPowerCorrelatedField(
        grid=grid,
        a=a,
    )
    sensitivity = jnp.ones(grid.shape)
    sensitivity = sensitivity.at[0:60, 50:-1].set(0)
    signal_response = SignalResponse(
        signal=signal,
        sensitivity=sensitivity,
    )

    def noise_cov(x):
        return (noise_std**2) * x

    def noise_cov_inv(x):
        return (noise_std**-2) * x

    key, sub_key = jax.random.split(key)
    pos_truth = jft.random_like(sub_key, signal_response.domain)
    signal_truth = signal(pos_truth)
    signal_response_truth = signal_response(pos_truth)

    fig = plt.figure()
    plt.imshow(
        signal_truth.T,
        origin="lower",
    )
    plt.title("Signal - Truth")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.show()

    key, sub_key = jax.random.split(key)
    noise_true = noise_std * jft.random_like(
        sub_key,
        signal_response.domain,
    )

    data = signal_response_truth + noise_true

    likelihood = jft.Gaussian(
        data,
        noise_cov_inv,
    ).amend(
        signal_response
    )  # `lh` in lecture notes

    fig = plt.figure()
    plt.imshow(
        data.T,
        origin="lower",
    )
    plt.title("Data")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.show()

    delta = 1e-6
    key, sub_key = jax.random.split(key)
    samples, info = jft.wiener_filter_posterior(
        likelihood,
        key=sub_key,
        n_samples=50,
        draw_linear_kwargs={
            "cg_name": "W",
            "cg_kwargs": {
                "absdelta": delta,
                "maxiter": 100,
            },
        },
    )
    post_mean, post_std = jft.mean_and_std(tuple(signal(s) for s in samples))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    im = ax.imshow(
        signal_truth.T,
        origin="lower",
    )
    ax.set_title("Ground Truth")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(
        data.T,
        origin="lower",
    )
    ax.set_title("Data")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    im = ax.imshow(
        post_mean.T,
        origin="lower",
    )
    ax.set_title("Posterior mean")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(
        post_std.T,
        origin="lower",
    )
    ax.set_title("Posterior std")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.suptitle("Day 1: Summary plot")
    plt.show()

    # --------------- Day 2 --------------- #
    signal = FixedPowerCorrelatedFieldNonLin(
        grid=grid,
        a=a,
    )
    signal_response = SignalResponse(
        signal=signal,
        sensitivity=sensitivity,
    )
    key, sub_key = jax.random.split(key)
    pos_truth = jft.random_like(sub_key, signal_response.domain)
    signal_truth = signal(pos_truth)
    signal_response_truth = signal_response(pos_truth)
    key, sub_key = jax.random.split(key)
    noise_true = noise_std * jft.random_like(
        sub_key,
        signal_response.domain,
    )

    data = signal_response_truth + noise_true

    likelihood = jft.Gaussian(
        data,
        noise_cov_inv,
    ).amend(
        signal_response
    )  # `lh` in lecture notes

    delta = 1e-6
    key, sub_key, sub_key2 = jax.random.split(key, 3)
    samples, info = jft.optimize_kl(
        likelihood,
        likelihood.init(sub_key),
        n_total_iterations=5,
        n_samples=4,
        key=sub_key2,
        draw_linear_kwargs={
            "cg_name": "cg linear sampling",
            "cg_kwargs": {
                "absdelta": delta,
                "maxiter": 100,
            },
        },
        kl_kwargs={
            "minimize_kwargs": {
                "name": "kl minimizer",
                "xtol": delta,
                "cg_kwargs": {
                    "name": None,
                },
            },
        },
        sample_mode="linear_resample",
    )

    post_mean, post_std = jft.mean_and_std(tuple(signal(s) for s in samples))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    im = ax.imshow(
        signal_truth.T,
        origin="lower",
    )
    ax.set_title("Ground Truth")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(
        data.T,
        origin="lower",
    )
    ax.set_title("Data")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    im = ax.imshow(
        post_mean.T,
        origin="lower",
    )
    ax.set_title("Posterior mean")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(
        post_std.T,
        origin="lower",
    )
    ax.set_title("Posterior std")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.suptitle("Day 2: Summary plot")
    plt.show()
