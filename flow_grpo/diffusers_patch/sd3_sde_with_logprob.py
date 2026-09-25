# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
import logging
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def _patchify(images, patch_size: int = 2):
    # Input Shape: [BS, C, H, W]
    # Output Shape: [BS, (H * W) // (patch_size * patch_size), C * patch_size * patch_size]
    BS, C, H, W = images.shape
    # [BS, C, H//p, p, W//p, p]
    x = images.reshape(BS, C, H // patch_size, patch_size, W // patch_size, patch_size)
    # [BS, H//p, W//p, C, p, p]
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(BS, -1, C * patch_size * patch_size) # Shape: [BS, seq_len, dim]


def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    cond_model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    uncond_model_output: torch.FloatTensor = None,
    guidance_scale = 1,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: Optional[str] = 'sde',
    return_sqrt_dt: Optional[bool] = False,
    reduction: bool = True,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma
    not_prev_sample = prev_sample is None

    if sde_type == 'sde':
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level
        noise_pred = uncond_model_output + guidance_scale * (cond_model_output - uncond_model_output)

        b_t = 1+std_dev_t**2*(1-sigma)/(2*sigma)
        tau = 0.01
        mse = (noise_pred - cond_model_output) ** 2
        mse_per_sample = mse.mean(dim=(1, 2, 3), keepdim=True)

        D_t_per_sample = mse_per_sample * ((b_t * dt).pow(2) / (2 * std_dev_t.pow(2) * (-dt) ))

        residual_scale = 1 - torch.exp(
            -D_t_per_sample / tau
        )
        
        # value_range = 0.9
        # residual_scale = residual_scale * value_range + 1 - value_range
        
        mix_cfg_pred = cond_model_output + residual_scale * (guidance_scale - 1) * (cond_model_output - uncond_model_output)
        prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+mix_cfg_pred*b_t*dt

        if not_prev_sample:
            variance_noise = randn_tensor(
                cond_model_output.shape,
                generator=generator,
                device=cond_model_output.device,
                dtype=cond_model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
            - torch.log(std_dev_t * torch.sqrt(-1*dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

        prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+noise_pred*b_t*dt

        if not_prev_sample:
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise
            
    elif sde_type == 'cps':
        std_dev_t = sigma_prev  * math.sin(noise_level * math.pi / 2) # sigma_t in paper
        pred_original_sample = sample - sigma * model_output # predicted x_0 in paper
        noise_estimate = sample + model_output * (1 - sigma) # predicted x_1 in paper
        prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        # remove all constants
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

    if reduction:
        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    else:
        # log_prob Shape: [BS, C, H, W]
        log_prob = _patchify(log_prob)
        log_prob = log_prob.sum(dim=-1) # Shape: [BS, seq_len]
    
    if return_sqrt_dt:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1*dt), residual_scale
    return prev_sample, log_prob, prev_sample_mean, std_dev_t, residual_scale