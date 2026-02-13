import os
import random
import uuid
import json
import time
from threading import Thread
from typing import Iterable

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoProcessor,
    TextIteratorStreamer,
)
from transformers.image_utils import load_image
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",  
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_800)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_500)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )
        
steel_blue_theme = SteelBlueTheme()

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.2em !important;
}

/* RadioAnimated Styles */
.ra-wrap{ width: fit-content; }
.ra-inner{
  position: relative; display: inline-flex; align-items: center; gap: 0; padding: 6px;
  background: var(--neutral-200); border-radius: 9999px; overflow: hidden;
}
.ra-input{ display: none; }
.ra-label{
  position: relative; z-index: 2; padding: 8px 16px;
  font-family: inherit; font-size: 14px; font-weight: 600;
  color: var(--neutral-500); cursor: pointer; transition: color 0.2s; white-space: nowrap;
}
.ra-highlight{
  position: absolute; z-index: 1; top: 6px; left: 6px;
  height: calc(100% - 12px); border-radius: 9999px;
  background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  transition: transform 0.2s, width 0.2s;
}
.ra-input:checked + .ra-label{ color: black; }

/* Dark mode adjustments for Radio */
.dark .ra-inner { background: var(--neutral-800); }
.dark .ra-label { color: var(--neutral-400); }
.dark .ra-highlight { background: var(--neutral-600); }
.dark .ra-input:checked + .ra-label { color: white; }

#gpu-duration-container {
    padding: 10px;
    border-radius: 8px;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    margin-top: 10px;
}
"""

MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

class RadioAnimated(gr.HTML):
    def __init__(self, choices, value=None, **kwargs):
        if not choices or len(choices) < 2:
            raise ValueError("RadioAnimated requires at least 2 choices.")
        if value is None:
            value = choices[0]

        uid = uuid.uuid4().hex[:8]
        group_name = f"ra-{uid}"

        inputs_html = "\n".join(
            f"""
            <input class="ra-input" type="radio" name="{group_name}" id="{group_name}-{i}" value="{c}">
            <label class="ra-label" for="{group_name}-{i}">{c}</label>
            """
            for i, c in enumerate(choices)
        )

        html_template = f"""
        <div class="ra-wrap" data-ra="{uid}">
          <div class="ra-inner">
            <div class="ra-highlight"></div>
            {inputs_html}
          </div>
        </div>
        """

        js_on_load = r"""
        (() => {
          const wrap = element.querySelector('.ra-wrap');
          const inner = element.querySelector('.ra-inner');
          const highlight = element.querySelector('.ra-highlight');
          const inputs = Array.from(element.querySelectorAll('.ra-input'));

          if (!inputs.length) return;

          const choices = inputs.map(i => i.value);

          function setHighlightByIndex(idx) {
            const n = choices.length;
            const pct = 100 / n;
            highlight.style.width = `calc(${pct}% - 6px)`;
            highlight.style.transform = `translateX(${idx * 100}%)`;
          }

          function setCheckedByValue(val, shouldTrigger=false) {
            const idx = Math.max(0, choices.indexOf(val));
            inputs.forEach((inp, i) => { inp.checked = (i === idx); });
            setHighlightByIndex(idx);

            props.value = choices[idx];
            if (shouldTrigger) trigger('change', props.value);
          }

          setCheckedByValue(props.value ?? choices[0], false);

          inputs.forEach((inp) => {
            inp.addEventListener('change', () => {
              setCheckedByValue(inp.value, true);
            });
          });
        })();
        """

        super().__init__(
            value=value,
            html_template=html_template,
            js_on_load=js_on_load,
            **kwargs
        )

def apply_gpu_duration(val: str):
    return int(val)

MODEL_ID_V = "nanonets/Nanonets-OCR2-3B"
processor_v = AutoProcessor.from_pretrained(MODEL_ID_V, trust_remote_code=True)
model_v = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_V,
    attn_implementation="kernels-community/flash-attn2",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_X = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"
processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True)
model_x = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X,
    attn_implementation="kernels-community/flash-attn2",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_A = "CohereForAI/aya-vision-8b"
processor_a = AutoProcessor.from_pretrained(MODEL_ID_A, trust_remote_code=True)
model_a = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID_A,
    attn_implementation="kernels-community/flash-attn2",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_W = "allenai/olmOCR-7B-0725"
processor_w = AutoProcessor.from_pretrained(MODEL_ID_W, trust_remote_code=True)
model_w = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_W,
    attn_implementation="kernels-community/flash-attn2",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

MODEL_ID_M = "reducto/RolmOCR"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M,
    attn_implementation="kernels-community/flash-attn2",
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

def calc_timeout_duration(model_name: str, text: str, image: Image.Image,
                          max_new_tokens: int, temperature: float, top_p: float,
                          top_k: int, repetition_penalty: float, gpu_timeout: int):
    """Calculate GPU timeout duration based on the last argument."""
    try:
        return int(gpu_timeout)
    except:
        return 60


@spaces.GPU(duration=calc_timeout_duration)
def generate_image(model_name: str, text: str, image: Image.Image,
                   max_new_tokens: int, temperature: float, top_p: float,
                   top_k: int, repetition_penalty: float, gpu_timeout: int):
    """
    Generates responses using the selected model for image input.
    Yields raw text and Markdown-formatted text.
    """
    if model_name == "RolmOCR-7B":
        processor = processor_m
        model = model_m
    elif model_name == "Qwen2-VL-OCR-2B":
        processor = processor_x
        model = model_x
    elif model_name == "Nanonets-OCR2-3B":
        processor = processor_v
        model = model_v
    elif model_name == "Aya-Vision-8B":
        processor = processor_a
        model = model_a
    elif model_name == "olmOCR-7B-0725":
        processor = processor_w
        model = model_w
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ]
    }]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True).to(device)

    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer


image_examples = [
    ["Perform OCR on the image precisely.", "examples/5.jpg"],
    ["Run OCR on the image and ensure high accuracy.", "examples/4.jpg"],
    ["Conduct OCR on the image with exact text recognition.", "examples/2.jpg"],
    ["Perform precise OCR extraction on the image.", "examples/1.jpg"],
    ["Convert this page to docling", "examples/3.jpg"],
]

with gr.Blocks() as demo:
    gr.Markdown("# **Multimodal OCR**", elem_id="main-title")
    with gr.Row():
        with gr.Column(scale=2):
            image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
            image_upload = gr.Image(type="pil", label="Upload Image", height=290)

            image_submit = gr.Button("Submit", variant="primary")
            gr.Examples(
                examples=image_examples,
                inputs=[image_query, image_upload]
            )
            
            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.7)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.1)
                
        with gr.Column(scale=3):
            gr.Markdown("## Output", elem_id="output-title")
            output = gr.Textbox(label="Raw Output Stream", interactive=True, lines=11)
            with gr.Accordion("(Result.md)", open=False):
                markdown_output = gr.Markdown(label="(Result.Md)")

            model_choice = gr.Radio(
                choices=["Nanonets-OCR2-3B", "olmOCR-7B-0725", "RolmOCR-7B",
                     "Aya-Vision-8B", "Qwen2-VL-OCR-2B"],
                label="Select Model",
                value="Nanonets-OCR2-3B"
            )
            
            with gr.Row(elem_id="gpu-duration-container"):
                with gr.Column():
                    gr.Markdown("**GPU Duration (seconds)**")
                    radioanimated_gpu_duration = RadioAnimated(
                        choices=["60", "90", "120", "180", "240"],
                        value="60",
                        elem_id="radioanimated_gpu_duration"
                    )
                    gpu_duration_state = gr.Number(value=60, visible=False)
            
            gr.Markdown("*Note: Higher GPU duration allows for longer processing but consumes more GPU quota.*")

    radioanimated_gpu_duration.change(
        fn=apply_gpu_duration, 
        inputs=radioanimated_gpu_duration, 
        outputs=[gpu_duration_state], 
        api_visibility="private"
    )

    image_submit.click(
        fn=generate_image,
        inputs=[model_choice, image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty, gpu_duration_state],
        outputs=[output, markdown_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(css=css, theme=steel_blue_theme, mcp_server=True, ssr_mode=False, show_error=True)
