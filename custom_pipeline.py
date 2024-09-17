import torch
import numpy as np
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from typing import Any, Dict, List, Optional, Union
from PIL import Image

# Constants for shift calculation
BASE_SEQ_LEN = 256
MAX_SEQ_LEN = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.2

# Helper functions
def calculate_timestep_shift(image_seq_len: int) -> float:
    """Calculates the timestep shift (mu) based on the image sequence length."""
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN)
    b = BASE_SHIFT - m * BASE_SEQ_LEN
    mu = image_seq_len * m + b
    return mu

def prepare_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    mu: Optional[float] = None,
) -> (torch.Tensor, int):
    """Prepares the timesteps for the diffusion process."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)

    timesteps = scheduler.timesteps
    num_inference_steps = len(timesteps)
    return timesteps, num_inference_steps

# FLUX pipeline function
class FLUXPipelineWithIntermediateOutputs(FluxPipeline):
    """
    Extends the FluxPipeline to yield intermediate images during the denoising process 
    with progressively increasing resolution for faster generation.
    """
    @torch.inference_mode()
    def generate_images(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 300,
    ):
        """Generates images and yields intermediate results during the denoising process."""
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device

        # 3. Encode prompt
        lora_scale = joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_timestep_shift(image_seq_len)
        timesteps, num_inference_steps = prepare_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        # Handle guidance
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float16).expand(latents.shape[0]) if self.transformer.config.guidance_embeds else None

        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

             # Yield intermediate result
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            torch.cuda.empty_cache()

        # Final image
        yield self._decode_latents_to_image(latents, height, width, output_type)
        self.maybe_free_model_hooks()
        torch.cuda.empty_cache()

    def _decode_latents_to_image(self, latents, height, width, output_type, vae=None):
        """Decodes the given latents into an image."""
        vae = vae or self.vae
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type=output_type)[0]
