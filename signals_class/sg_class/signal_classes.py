from functools import partial
from typing import Optional

import jax
import nifty8.re as jft
from jax import numpy as jnp
from jax.typing import ArrayLike

jax.config.update("jax_enable_x64", True)


class FixedPowerCorrelatedField(jft.Model):
    r"""Fixed power law correlated field."""

    def __init__(
        self,
        grid: jft.correlated_field.RegularCartesianGrid,
        a: ArrayLike,
    ):
        self.grid = grid
        self.a = a
        super().__init__(
            domain=jax.ShapeDtypeStruct(
                shape=grid.shape,
                dtype=jnp.float64,
            ),
        )

    def __call__(self, x):
        ht = jft.correlated_field.hartley
        harmonic_dvol = 1 / self.grid.total_volume
        return harmonic_dvol * ht(self.a * x)


class FixedPowerCorrelatedFieldNonLin(FixedPowerCorrelatedField):
    r"""Fixed power law correlated field."""

    def __call__(self, x):
        gaussian = super().__call__(x)
        return jnp.exp(gaussian)


class SignalResponse(jft.Model):
    r"""Signal response."""

    def __init__(
        self,
        signal: FixedPowerCorrelatedField,
        sensitivity: ArrayLike,
    ):
        self.signal = signal
        self.sensitivity = sensitivity
        super().__init__(
            domain=signal.domain,
        )

    def __call__(self, x):
        return self.signal(x) * self.sensitivity


class SignalCF(jft.Model):
    def __init__(
        self,
        correlated_field: jft.model.Model,
    ):
        self.cf = correlated_field
        super().__init__(init=self.cf.init)

    def __call__(
        self,
        x,
    ):
        return jnp.exp(self.cf(x))


class SignalCFRemovePadding(jft.Model):
    def __init__(
        self,
        field: jft.model.Model,
        pad_left: int,
        pad_right: Optional[int] = None,
    ):
        self.filed = field
        self.pad_left = pad_left
        self.pad_right = pad_right if pad_right is not None else pad_left
        super().__init__(init=self.filed.init)

    def __call__(
        self,
        x,
    ):
        res = self.filed(x)
        res = res[self.pad_left : -self.pad_right]
        return res


class SignalBase(jft.Model):
    def __init__(
        self,
        correlated_field: jft.model.Model,
        pad_left: int,
        pad_right: Optional[int] = None,
    ):
        self.cf = correlated_field
        self.pad_left = pad_left
        self.pad_right = pad_right if pad_right is not None else pad_left
        super().__init__(init=self.cf.init)

    def __call__(
        self,
        x,
    ):
        res = jnp.exp(self.cf(x))
        res = res[self.pad_left : -self.pad_right]
        return res


class StonksModel(jft.Model):
    def __init__(
        self,
        nt: int,
        dt: float = 1.0,
        name: str = "",
        avg_yearly_yield: float = 1.05,
        std_yearly_yield: float = 0.05,
        mean_fluctuation_amplitude: float = 0.20,
        std_fluctuation_amplitude: float = 0.05,
        ohp_gamma: float = 0.1,
        starting_price_mean: float = 3000.0,
        starting_price_std: float = 1.0,
    ):
        self.nt = nt
        self.t = jnp.arange(self.nt)
        self.dt = dt
        self.name = name

        self.yearly_yield = jft.LogNormalPrior(
            mean=avg_yearly_yield,
            std=std_yearly_yield,
            name=f"{self.name}_yearly_yield",
        )
        self.gp_fluctuations = jft.OrnsteinUhlenbeckProcess(
            sigma=jft.LogNormalPrior(
                mean=mean_fluctuation_amplitude / 3,
                std=std_fluctuation_amplitude / 3,
                name=f"{self.name}_fluctuation_amplitude",
            ),
            gamma=ohp_gamma,
            dt=self.dt,
            N_steps=self.nt - 1,
            x0=0.0,
        )
        self.starting_price = jft.LogNormalPrior(
            mean=starting_price_mean,
            std=starting_price_std,
            name=f"{self.name}_starting_price",
        )
        super().__init__(
            init=partial(
                jft.random_like,
                primals=self.yearly_yield.domain.copy()
                | self.gp_fluctuations.domain
                | self.starting_price.domain,
            )
        )

    def __call__(self, x):
        starting_price_val = self.starting_price(x)

        yearly_yield = self.yearly_yield(x)
        daily_yield = jnp.float_power(yearly_yield, 1 / 365)
        log_price_slope = jnp.log(daily_yield)
        log_price_pred = log_price_slope * self.t

        price_fluctuations = self.gp_fluctuations(x)

        return starting_price_val * jnp.exp(log_price_pred) * (1.0 + price_fluctuations)


# class SignalRes(jft.Model):
#     def __call__(self, x):
#         res = super().__call__(x)[self.data_size:]
#         return res


# class SignalRes(jft.Model):
#     def __call__(self, x):
#         res = super().__call__(x)[self.data_size:]
#         return res
