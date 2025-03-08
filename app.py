import gradio as gr
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
    AutoModelForImageTextToText,
)
from transformers.image_utils import load_image
from threading import Thread
import time
import torch
import spaces
from PIL import Image
import requests
from io import BytesIO

# Helper function to return a progress bar HTML snippet.
def progress_bar_html(label: str) -> str:
    return f'''
<div style="display: flex; align-items: center;">
    <span style="margin-right: 10px; font-size: 14px;">{label}</span>
    <div style="width: 110px; height: 5px; background-color: #98FB98; border-radius: 2px; overflow: hidden;">
        <div style="width: 100%; height: 100%; background-color: #00FF7F; animation: loading 1.5s linear infinite;"></div>
    </div>
</div>
<style>
@keyframes loading {{
    0% {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(100%); }}
}}
</style>
    '''

QV_MODEL_ID = "prithivMLmods/Qwen2-VL-OCR2-2B-Instruct" # or use  #prithivMLmods/Qwen2-VL-OCR-2B-Instruct
qwen_processor = AutoProcessor.from_pretrained(QV_MODEL_ID, trust_remote_code=True)
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    QV_MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda").eval()

AYA_MODEL_ID = "CohereForAI/aya-vision-8b"
aya_processor = AutoProcessor.from_pretrained(AYA_MODEL_ID)
aya_model = AutoModelForImageTextToText.from_pretrained(
    AYA_MODEL_ID, device_map="auto", torch_dtype=torch.float16
)

@spaces.GPU
def model_inference(input_dict, history):
    text = input_dict["text"].strip()
    files = input_dict.get("files", [])
    
    if text.lower().startswith("@aya-vision"):
        # Remove the command prefix and trim the prompt.
        text_prompt = text[len("@aya-vision"):].strip()
        if not files:
            yield "Error: Please provide an image for the @aya-vision feature."
            return
        else:
            # For simplicity, use the first provided image.
            image = load_image(files[0])
            yield progress_bar_html("Processing with Aya-Vision")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }]
            inputs = aya_processor.apply_chat_template(
                messages,
                padding=True,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(aya_model.device)
            # Set up a streamer for Aya-Vision output
            streamer = TextIteratorStreamer(aya_processor, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                inputs, 
                streamer=streamer, 
                max_new_tokens=1024, 
                do_sample=True, 
                temperature=0.3
            )
            thread = Thread(target=aya_model.generate, kwargs=generation_kwargs)
            thread.start()
            buffer = ""
            for new_text in streamer:
                buffer += new_text
                buffer = buffer.replace("<|im_end|>", "")
                time.sleep(0.01)
                yield buffer
            return

    # Load images if provided.
    if len(files) > 1:
        images = [load_image(image) for image in files]
    elif len(files) == 1:
        images = [load_image(files[0])]
    else:
        images = []
    
    # Validate input: require both text and (optionally) image(s).
    if text == "" and not images:
        yield "Error: Please input a query and optionally image(s)."
        return
    if text == "" and images:
        yield "Error: Please input a text query along with the image(s)."
        return

    # Prepare messages for the Qwen2-VL model.
    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": image} for image in images],
            {"type": "text", "text": text},
        ],
    }]
    
    prompt = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = qwen_processor(
        text=[prompt],
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    
    # Set up a streamer for real-time output.
    streamer = TextIteratorStreamer(qwen_processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)
    
    # Start generation in a separate thread.
    thread = Thread(target=qwen_model.generate, kwargs=generation_kwargs)
    thread.start()
    
    buffer = ""
    yield progress_bar_html("Processing with Qwen2VL OCR")
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer

examples = [
    [{"text": "@aya-vision Extract as JSON table from the table", "files": ["examples/4.jpg"]}],
    [{"text": "@aya-vision Extract JSON from the image", "files": ["example_images/document.jpg"]}],
    [{"text": "@aya-vision Summarize the letter", "files": ["examples/1.png"]}],
    [{"text": "@aya-vision Describe the photo", "files": ["examples/3.png"]}],
    [{"text": "@aya-vision Summarize the full image in detail", "files": ["examples/2.jpg"]}],
    [{"text": "@aya-vision Describe this image.", "files": ["example_images/campeones.jpg"]}],
    [{"text": "@aya-vision What is this UI about?", "files": ["example_images/s2w_example.png"]}],
    [{"text": "Can you describe this image?", "files": ["example_images/newyork.jpg"]}],
    [{"text": "Can you describe this image?", "files": ["example_images/dogs.jpg"]}],
    [{"text": "@aya-vision Where do the severe droughts happen according to this diagram?", "files": ["example_images/examples_weather_events.png"]}],
]

demo = gr.ChatInterface(
    fn=model_inference,
    description="# **Multimodal OCR `@aya-vision 'prompt..'`**",
    examples=examples,
    textbox=gr.MultimodalTextbox(
        label="Query Input", 
        file_types=["image"], 
        file_count="multiple", 
        placeholder="By default, it runs Qwen2VL OCR, Tag @aya-vision for Aya Vision 8B"
    ),
    stop_btn="Stop Generation",
    multimodal=True,
    cache_examples=False,
)

demo.launch(debug=True)
