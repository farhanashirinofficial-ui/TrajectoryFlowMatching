import torch

class SDE_func_solver(torch.nn.Module):

    noise_type = "diagonal"
    sde_type = "ito"

    # noise is sigma in this notebook for the equation sigma * (t * (1 - t))
    def __init__(self, ode_drift, noise, reverse=False, diagnostic_callback=None):
        super().__init__()
        self.drift = ode_drift
        self.reverse = reverse
        self.noise = noise # changeable, a model itself
        self.diagnostic_callback = diagnostic_callback
        self.diagnostic_context = {}
        self.last_diagnostic_tensors = {}

    # Drift
    def f(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        return self.drift(x)

    # Diffusion
    def g(self, t, y):
        if self.reverse:
            t = 1 - t
        if len(t.shape) == len(y.shape):
            x = torch.cat([y, t], 1)
        else:
            x = torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)
        noise_result = self.noise(x)
        time_factor = t * (1 - t)
        scheduled_diffusion = noise_result * torch.sqrt(time_factor)
        if self.diagnostic_callback is not None:
            tensors = {
                "raw_t": t,
                "t_times_one_minus_t": time_factor,
                "raw_learned_diffusion": noise_result,
                "scheduled_diffusion": scheduled_diffusion,
            }
            self.last_diagnostic_tensors = tensors
            if any(not torch.isfinite(value).all() for value in tensors.values()):
                self.diagnostic_callback(
                    "SDE_func_solver.g", tensors, self.diagnostic_context
                )
        return scheduled_diffusion
