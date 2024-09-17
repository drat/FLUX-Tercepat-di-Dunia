import gradio as gr
import numpy as np
import random
import spaces
import torch
import time
from diffusers import DiffusionPipeline
from custom_pipeline import FLUXPipelineWithIntermediateOutputs

# Constants
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_INFERENCE_STEPS = 1

# Device and model setup
dtype = torch.float16
pipe = FLUXPipelineWithIntermediateOutputs.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=dtype
).to("cuda")
torch.cuda.empty_cache()

# Inference function
@spaces.GPU(duration=25)
def generate_image(prompt, seed=42, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, randomize_seed=False, num_inference_steps=2, progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(int(float(seed)))

    start_time = time.time()

    # Only generate the last image in the sequence
    for img in pipe.generate_images(  
            prompt=prompt,
            guidance_scale=0, # as Flux schnell is guidance free
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator
        ): 
        latency = f"Latency: {(time.time()-start_time):.2f} seconds"    
        yield img, seed, latency

# Example prompts
examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cute white cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
    "Create mage of Modern house in minecraft style",
    "Imagine steve jobs as Star Wars movie character",
    "Lion",
    "Photo of a young woman with long, wavy brown hair tied in a bun and glasses. She has a fair complexion and is wearing subtle makeup, emphasizing her eyes and lips. She is dressed in a black top. The background appears to be an urban setting with a building facade, and the sunlight casts a warm glow on her face.",
]

# --- Gradio UI ---
with gr.Blocks() as demo:
    with gr.Column(elem_id="app-container"):
        gr.Markdown("# 🎨 Realtime FLUX Image Generator")
        gr.Markdown("Generate stunning images in real-time with Modified Flux.Schnell pipeline.")
        gr.Markdown("<span style='color: red;'>Note: Sometimes it stucks or stops generating images (I don't know why). In that situation just refresh the site.</span>")

        with gr.Row():
            with gr.Column(scale=2.5):
                result = gr.Image(label="Generated Image", show_label=False, interactive=False)
            with gr.Column(scale=1):
                prompt = gr.Text(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                    show_label=False,
                    container=False,
                )
                generateBtn = gr.Button("🖼️ Generate Image")
                enhanceBtn = gr.Button("🚀 Enhance Image")

                with gr.Column("Advanced Options"):
                    with gr.Row():
                        realtime = gr.Checkbox(label="Realtime Toggler", info="If TRUE then uses more GPU but create image in realtime.", value=False)
                        latency = gr.Text(label="Latency")
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=42)
                        randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)
                    with gr.Row():
                        width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=DEFAULT_WIDTH)
                        height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=DEFAULT_HEIGHT)
                        num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=4, step=1, value=DEFAULT_INFERENCE_STEPS)

        with gr.Row():
            gr.Markdown("### 🌟 Inspiration Gallery")
        with gr.Row():
            gr.Examples(
                examples=examples,
                fn=generate_image,
                inputs=[prompt],
                outputs=[result, seed, latency],
                cache_examples="lazy" 
            )

    def enhance_image(*args):
        gr.Info("Enhancing Image") # currently just runs optimized pipeline for 2 steps. Further implementations later.
        return next(generate_image(*args))

    enhanceBtn.click(
        fn=enhance_image,
        inputs=[prompt, seed, width, height],
        outputs=[result, seed, latency],
        show_progress="hidden",
        api_name="Enhance",
        queue=False,
        concurrency_limit=None
    )

    generateBtn.click(
        fn=generate_image,
        inputs=[prompt, seed, width, height, randomize_seed, num_inference_steps],
        outputs=[result, seed, latency],
        show_progress="full",
        api_name="RealtimeFlux",
        queue=False,
        concurrency_limit=None
    )

    def update_ui(realtime_enabled):
        return {
            prompt: gr.update(interactive=True),
            generateBtn: gr.update(visible=not realtime_enabled)
        }

    realtime.change(
        fn=update_ui,
        inputs=[realtime],
        outputs=[prompt, generateBtn],
        queue=False,
        concurrency_limit=None
    )

    def realtime_generation(*args):
        if args[0]:  # If realtime is enabled
            return next(generate_image(*args[1:]))

    prompt.submit(
        fn=generate_image,
        inputs=[prompt, seed, width, height, randomize_seed, num_inference_steps],
        outputs=[result, seed, latency],
        show_progress="full",
        api_name=False,
        queue=False,
        concurrency_limit=None
    )

    for component in [prompt, width, height, num_inference_steps]:
        component.input(
            fn=realtime_generation,
            inputs=[realtime, prompt, seed, width, height, randomize_seed, num_inference_steps],
            outputs=[result, seed, latency],
            show_progress="hidden",
            api_name=False,
            trigger_mode="always_last",
            queue=False,
            concurrency_limit=None
        )

# Launch the app
demo.launch()
