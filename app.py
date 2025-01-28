import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from transformers.image_utils import load_image
from threading import Thread
import time
import torch
import spaces

# Fine-tuned for OCR-based tasks from Qwen's [ Qwen/Qwen2-VL-2B-Instruct ]
MODEL_ID = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct" 
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda").eval()

@spaces.GPU
def model_inference(input_dict, history):
    text = input_dict["text"]
    files = input_dict["files"]

    # Load images if provided
    if len(files) > 1:
        images = [load_image(image) for image in files]
    elif len(files) == 1:
        images = [load_image(files[0])]
    else:
        images = []

    # Validate input
    if text == "" and not images:
        gr.Error("Please input a query and optionally image(s).")
        return
    if text == "" and images:
        gr.Error("Please input a text query along with the image(s).")
        return

    # Prepare messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": text},
            ],
        }
    ]

    # Apply chat template and process inputs
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Set up streamer for real-time output
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)

    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the output
    buffer = ""
    yield "Thinking..."
    for new_text in streamer:
        buffer += new_text
        # Remove <|im_end|> or similar tokens from the output
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer

# Example inputs
examples = [

    [{"text": "Extract JSON from the image", "files": ["example_images/document.jpg"]}],
    [{"text": "summarize the letter", "files": ["examples/1.png"]}],
    [{"text": "Describe the photo", "files": ["examples/3.png"]}],
    [{"text": "Extract as JSON table from the table", "files": ["examples/4.jpg"]}],
    [{"text": "Summarize the full image in detail", "files": ["examples/2.jpg"]}],
    [{"text": "Describe this image.", "files": ["example_images/campeones.jpg"]}],
    [{"text": "What is this UI about?", "files": ["example_images/s2w_example.png"]}],
    [{"text": "Can you describe this image?", "files": ["example_images/newyork.jpg"]}],
    [{"text": "Can you describe this image?", "files": ["example_images/dogs.jpg"]}],
    [{"text": "Where do the severe droughts happen according to this diagram?", "files": ["example_images/examples_weather_events.png"]}],

]

demo = gr.ChatInterface(
    fn=model_inference,
    description="# **Multimodal OCR**",
    examples=examples,
    textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image"], file_count="multiple"),
    stop_btn="Stop Generation",
    multimodal=True,
    cache_examples=False,
)

demo.launch(debug=True)