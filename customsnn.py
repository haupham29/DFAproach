import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax
from typing import Callable, Optional
from diffrax import Solution
import optimistix as optx

jax.config.update("jax_platform_name", "cpu")

class SpikingNeuron(eqx.Module):
    """Single-neuron SNN with:
       - mu, sigma for (v, i)
       - a spike variable s: ds/dt = intensity(v)
    """
    mu: jnp.ndarray            
    sigma: jnp.ndarray         
    intensity_params: jnp.ndarray
    alpha: float              
    v_reset: float

    def intensity(self, v):
        """Example: λ(v) = exp(k*(v - threshold)) From the Paper"""
        k = self.intensity_params[0]
        threshold = 1.0
        return jnp.exp(k*(v - threshold))

    def drift(self, t, y, args):
        """y = (v, i, s). dv/dt = mu1*(i - v), di/dt = -mu2*i, ds/dt = intensity(v)."""
        v, i, s = y
        mu1, mu2 = self.mu
        dv = mu1*(i - v)
        di = -mu2*i
        ds = self.intensity(v)
        return jnp.stack([dv, di, ds])

    def diffusion(self, t, y, args):
        """Inject noise into (v, i). s has no noise => last row=0."""
        if self.sigma is None:
            return jnp.zeros((3,2))
        
        block = jnp.array([
            [self.sigma[0,0], self.sigma[0,1]],
            [self.sigma[1,0], self.sigma[1,1]],
            [0.0,              0.0]
        ])
        return block

    def event_fn(self, y):
        """spike when s crosses 0 from below."""
        return y[2]

    def reset_map(self, y, rng_key):
        v, i, s = y
        u = jax.random.uniform(rng_key, shape=())
        new_v = v - self.v_reset
        new_s = jnp.log(u) - self.alpha
        return jnp.array([new_v, i, new_s])

def solve_spike(neuron: SpikingNeuron,
                y0: jnp.ndarray,
                t0: float, t1: float,
                dt0: float,
                rng_key,
                ) -> tuple[float, jnp.ndarray]:
    # Create the drift/diffusion terms
    drift_term = diffrax.ODETerm(neuron.drift)
    if neuron.sigma is not None:
        # Brownian path needed for SDE
        bm = diffrax.VirtualBrownianTree(t0, t1, shape=(2,), key=rng_key, tol=1e-6)
        diffusion_term = diffrax.ControlTerm(neuron.diffusion, bm)
        terms = diffrax.MultiTerm(drift_term, diffusion_term)
    else:
        # no diffusion => ODE
        terms = drift_term

    # We'll define an event that triggers when s == 0
    def event_cond(t, y, args, **kwargs):
        # Check if y is None and handle it
        if y is None:
            return jnp.array(float('inf'))  # No event if y is None
        return neuron.event_fn(y)

    # Create the event without the terminal parameter
    root_finder = optx.Newton(1e-2, 1e-2, optx.rms_norm)
    event = diffrax.Event(event_cond, root_finder=root_finder) 

    saveat = diffrax.SaveAt(t0=False, t1=True)
    solver = diffrax.EulerHeun() 
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=None,
        saveat=saveat,
        event=event,          # events=Event(...) 
        max_steps=10_000,
    )
    # Fix: The event detection should look for t_final < t1
    # If solver stopped early, it means an event was detected
    is_spike = sol.ts[-1] < t1 - 1e-6
    spike_time = jnp.where(is_spike, sol.ts[-1], t1)
    # No need for conditional here since we return the state at the end time either way
    return spike_time, sol.ys[-1]

def run_spiking_neuron(neuron: SpikingNeuron,
                       y0: jnp.ndarray,
                       t0: float, t1: float,
                       dt0: float,
                       rng_key,
                       max_spikes=5):
    """
    Solve from t0 to t1, stopping each time a spike event triggers,
    applying the reset map, and continuing.
    """
    times = jnp.zeros(max_spikes)
    states = jnp.zeros((max_spikes, 3))  # Assuming state is 3-dimensional
    
    curr_t = t0
    curr_y = y0
    
    for i in range(max_spikes):
        subkey, subkey_reset = jax.random.split(rng_key)
        rng_key = subkey
        
        spike_t, spike_y = solve_spike(neuron, curr_y, curr_t, t1, dt0, subkey_reset)
        
        # Store results (even if no spike occurred)
        times = times.at[i].set(spike_t)
        states = states.at[i].set(spike_y)
        
        # Determine if we should continue (no break statements)
        continue_sim = spike_t < t1 - 1e-12
        
        # Apply reset
        post_spike_y = jax.lax.cond(
            continue_sim,
            lambda: neuron.reset_map(spike_y, subkey_reset),
            lambda: curr_y
        )
        
        # Update for next iteration (happens regardless of spike)
        curr_t = jax.lax.cond(continue_sim, lambda: spike_t, lambda: curr_t)
        curr_y = post_spike_y
    
    return times, states

# --------------------------
#  Example usage
# --------------------------

def main():
    neuron = SpikingNeuron(
        mu=jnp.array([5.0, 3.0]),
        sigma=jnp.array([[0.2, 0.0],
                         [0.0, 0.2]]),
        intensity_params=jnp.array([5.0]),
        alpha=0.03,
        v_reset=1.0,
    )

    # initial state 
    y0 = jnp.array([0.2, 0.9, jnp.log(0.2) - neuron.alpha])

    # Solve from t=0 to t=3
    times, states = run_spiking_neuron(neuron, y0,
                                       t0=0.0, t1=10.0,
                                       dt0=0.01,
                                       rng_key=jax.random.PRNGKey(777),
                                       max_spikes=10)
    print("Spike times:", times)
    print("States:", states)

    import optax 
    def loss_fn(params: SpikingNeuron, key):
        
        # 1) Run the spiking simulation with the given parameters.
        # 2) Compare the first spike time to a target, e.g. 1.5s.
        # 3) Return a scalar loss.
        
        y0 = jnp.array([0.5, 0.2, jnp.log(0.5) - params.alpha])  # initial (v, i, s)

        times, states = run_spiking_neuron(
            neuron=params,
            y0=y0,
            t0=0.0, t1=3.0,
            dt0=0.01,
            rng_key=key,
            max_spikes=1,  # We only care about first spike for now
        )
        
        # Check if there was a spike before t1
        t_spike = times[0]
        t_desired = 1.5
        return (t_spike - t_desired)**2

    def train_step(params: SpikingNeuron, opt_state, optimizer, rng):
        """Perform one gradient step."""
        # Compute loss and grads
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, rng)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = eqx.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @eqx.filter_jit
    def train_step(params, opt_state, optimizer, key):
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss_val

    def train():
        # Initialize neuron with parameters we want to train
        params = SpikingNeuron(
            mu=jnp.array([5.0, 3.0]),
            sigma=jnp.array([[0.2, 0.0],[0.0, 0.2]]),
            intensity_params=jnp.array([5.0]),
            alpha=0.03,
            v_reset=1.0,
        )
        
        learning_rate = 0.1
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(eqx.filter(params, eqx.is_array))

        key = jr.PRNGKey(1234)

        for step in range(100):
            key, subkey = jr.split(key)
            # Use jit to speed up training
            loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params, subkey)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = eqx.apply_updates(params, updates)
            
            if step % 10 == 0:
                print(f"Step={step}, Loss={loss_val}")
                print(f"mu={params.mu}, intensity_params={params.intensity_params}, alpha={params.alpha}")

        print("Final learned params:")
        print(params)

    train()
if __name__ == "__main__":    main()
