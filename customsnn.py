import jax.random as jr
from diffrax import diffeqsolve, ControlTerm, Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Callable, List, Optional
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, Real
import optimistix as optx
import diffrax
import functools as ft



class CustomStochastic(eqx.Module):
    mu: jnp.ndarray            
    sigma: jnp.ndarray         
    intensity_fn: Callable[..., Float]          
    v_reset: float
    threshold: float
    alpha: float
    cond_fn: List[Callable[..., Float]]
    drift_vf: Callable[..., Float]
    diffusion_vf: Callable[..., Float]

    def __init__(
        self, mu, sigma, intensity_fn, v_reset=1.0, threshold=1.0, alpha=3e-2):
        """**Arguments**:

        - `intensity_fn`: The intensity function for spike generation.
            Should take as input a scalar (voltage) and return a scalar (intensity).
        - `v_reset`: The reset voltage value for neurons. Defaults to 1.0.
        - `alpha`: Constant controlling the refractory period. Defaults to 3e-2.
        - `mu`: A 2-dimensional vector describing the drift term of each neuron.
            If none is provided, the values are randomly initialized.
        - `sigma`: A 2 by 2 diffusion matrix. If none is provided, the values are randomly
            initialized..
        """
        self.mu = mu
        self.threshold = threshold
        
        def intensity_fn(v):
            return jnp.maximum(0, jnp.exp(v) * (v - self.threshold))
        self.intensity_fn = intensity_fn
        self.v_reset = v_reset
        self.alpha = alpha
        if sigma is None:
            sigma_key = jr.PRNGKey(0)
            sigma = jr.normal(sigma_key, (2, 2))
            sigma = jnp.dot(sigma, sigma.T)
        self.sigma = sigma

        def cond_fn( t, y, args, **kwargs):
            v, i, s = y
            return s

        self.cond_fn = cond_fn

        def drift_vf(t, y, args):
            v, i, s = y
            mu1, mu2 = self.mu
            ic = args  # args is the input_current
            dv = mu1 * (i + ic - v)
            di = -mu2 * i
            ds = self.intensity_fn(v)
            return jnp.array([dv, di, ds])
        self.drift_vf = drift_vf
        
        def diffusion_vf(t, y, args):
            full_sigma = jnp.zeros((3, 3))
            full_sigma = full_sigma.at[:2, :2].set(self.sigma)
            return full_sigma
        self.diffusion_vf = diffusion_vf

    @eqx.filter_jit
    def __call__(self,
            input_current,
            t0,
            t1,
            v0=None,
            i0=None,
            dt0=0.01,
            max_steps=1000
        ):
        s0_key, v0_key, i0_key = jr.split(jr.PRNGKey(159), 3)
        bm_key = jr.PRNGKey(9876)

        s0 = jnp.log(jr.uniform(key=s0_key)) - self.alpha
        if v0 is None:
            v0 = jr.uniform(key=v0_key)
        if i0 is None:
            i0 = jr.uniform(key=i0_key)
        y0 = jnp.array([v0, i0, s0])

        key = jr.PRNGKey(1234)
        vf = ODETerm(self.drift_vf)

        bm = VirtualBrownianTree(t0, t1, 1e-3, (3, ), key=bm_key)
        diffusion = ControlTerm(self.diffusion_vf, bm)
        root_finder = optx.Newton(1e-2, 1e-2, optx.rms_norm)
        # Event detection is when s crosses 0
        event = diffrax.Event(self.cond_fn, root_finder)
        
        solver = diffrax.Euler() 
        
        sde = MultiTerm(vf, diffusion)

        sol = diffrax.diffeqsolve(
            sde,
            solver,
            t0, 
            t1,
            dt0,
            y0,
            input_current,
            throw=True,
            event=event,
            max_steps=max_steps,
            )
        
        # Check if we got a spike (event occurred)
        final_time = sol.ts[-1]
        final_state = sol.ys[-1]
        # Check if integration stopped early due to event
        spike_occurred = final_time < (t1 - 1e-7)
        # Also check state to confirm it was an event trigger
        spike_time = jnp.where(spike_occurred, final_time, jnp.nan)
        return sol, spike_time