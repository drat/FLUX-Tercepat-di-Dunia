import gradio as gr
import random
from gradio_client import Client
import os
from themes import IndonesiaTheme  # Impor tema custom dari themes.py

# Constants
MAX_SEED = 999999
MAX_IMAGE_SIZE = 2048
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_INFERENCE_STEPS = 1

# Siapkan URL untuk permintaan API RT FLUX
# url_api = os.environ['url_api']

client = Client("KingNish/Realtime-FLUX")
# client = Client(url_api)

# Inference function using RealtimeFlux API
def generate_image(prompt, seed=42, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, randomize_seed=False, num_inference_steps=1):
    result = client.predict(
        prompt=prompt,
        seed=seed if not randomize_seed else random.randint(0, MAX_SEED),
        width=width,
        height=height,
        randomize_seed=randomize_seed,
        num_inference_steps=num_inference_steps,
        api_name="/RealtimeFlux"
    )
    return result[0], result[1], result[2]  # Image, Seed, Latency

# Enhance function using Enhance API
def enhance_image(prompt, seed, width, height):
    result = client.predict(
        param_0=prompt,
        param_1=seed,
        param_2=width,
        param_3=height,
        api_name="/Enhance"
    )
    return result[0], result[1], result[2]  # Image, Seed, Latency

# CSS untuk styling antarmuka
css = """
#col-left, #col-mid, #col-right {
    margin: 0 auto;
    max-width: 400px;
    padding: 10px;
    border-radius: 15px;
    background-color: #f9f9f9;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

#col-right {
    margin: 0 auto;
    max-width: 800px;
    padding: 10px;
    border-radius: 15px;
    background-color: #f9f9f9;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

#banner {
    width: 100%;
    text-align: center;
    margin-bottom: 20px;
}
#run-button {
    background-color: #ff4b5c;
    color: white;
    font-weight: bold;
    padding: 10px;
    border-radius: 10px;
    cursor: pointer;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
#footer {
    text-align: center;
    margin-top: 20px;
    color: silver;
}

#whitey {
    text-align: center;
    margin-top: 10px;
    color: white;
}
"""

# Membuat antarmuka Gradio dengan tema IndonesiaTheme
with gr.Blocks(css=css, theme=IndonesiaTheme()) as RealtimeFluxAPP:
    # Tambahkan banner dan header
    gr.HTML("""
        <div style='text-align: center;'>
            <h1>🌟 Realtime FLUX Image Generator 🌟</h1>
            <p>Selamat datang! Buat gambar memukau secara realtime dengan FLUX Pipeline.</p>
            <img src='https://i.ibb.co.com/M2Sd185/banner-rtf.jpg' alt='Banner' style='width: 100%; height: auto;'/>
        </div>
    """)

    # Layout utama
    with gr.Row():
        with gr.Column(elem_id="col-left"):
            gr.Markdown("### Deskripsi Gambar")
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Deskripsikan gambar yang ingin Anda buat...",
                lines=3,
                show_label=False,
                container=False,
            )
            generateBtn = gr.Button("🖼️ Buat Gambar", elem_id="run-button")
            enhanceBtn = gr.Button("🚀 Tingkatkan Gambar", elem_id="run-button")
            
            # Advanced Options
            with gr.Accordion("Advanced Options"):
                with gr.Row():
                    realtime = gr.Checkbox(label="Realtime Toggler", info="Gunakan lebih banyak GPU untuk realtime.")
                    latency = gr.Textbox(label="Latency")
                with gr.Row():
                    seed = gr.Number(label="Seed", value=42)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=DEFAULT_WIDTH)
                    height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=32, value=DEFAULT_HEIGHT)
                    num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=4, step=1, value=DEFAULT_INFERENCE_STEPS)
        
        # Output di sebelah kanan
        with gr.Column(elem_id="col-right"):
            result = gr.Image(label="Hasil Gambar", show_label=False, interactive=False)

    # Example Gallery
    gr.Markdown("### 🌟 Inspirasi Gallery")
    gr.Examples(elem_id="whitey",
        examples=[
            "A beautiful sunset over the rice fields in Bali",
            "A traditional Indonesian fisherman sailing in a wooden boat",
            "Mount Bromo erupting at dawn with the sky full of stars",
            "A street food vendor selling nasi goreng in Jakarta",
            "The majestic Komodo dragon walking through the forest",
            "An Indonesian traditional dancer performing in a colorful costume",
            "A futuristic cityscape of Jakarta with skyscrapers and advanced technology",
        ],
        fn=generate_image,
        inputs=[prompt],
        outputs=[result, seed, latency],
        cache_examples="lazy"
    )

    # Tombol untuk memulai proses pembuatan dan peningkatan gambar
    generateBtn.click(
        fn=generate_image,
        inputs=[prompt, seed, width, height, randomize_seed, num_inference_steps],
        outputs=[result, seed, latency],
        show_progress=True
    )

    enhanceBtn.click(
        fn=enhance_image,
        inputs=[prompt, seed, width, height],
        outputs=[result, seed, latency],
        show_progress=True
    )

    # Tambahkan footer di bagian bawah
    gr.HTML("""
    <footer id="footer">
        <p>Transfer Energi Semesta Digital © 2024 __drat. | 🇮🇩 Untuk Indonesia Jaya!</p>
    </footer>
    """)

# Menjalankan aplikasi
if __name__ == "__main__":
    RealtimeFluxAPP.queue(api_open=False).launch(show_api=False)
