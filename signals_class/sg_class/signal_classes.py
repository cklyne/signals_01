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


# class SignalRes(jft.Model):
#     def __call__(self, x):
#         res = super().__call__(x)[self.data_size:]
#         return res


# class SignalRes(jft.Model):
#     def __call__(self, x):
#         res = super().__call__(x)[self.data_size:]
#         return res
