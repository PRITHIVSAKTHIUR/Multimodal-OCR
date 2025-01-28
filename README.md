---
title: OCR
emoji: 🍍
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.13.1
app_file: app.py
pinned: true
license: creativeml-openrail-m
short_description: Qwen VL 2B
---
# Qwen2-VL-OCR-2B-Instruct [ VL / OCR ]

![aaaaaaaaaaa.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/s42kASSQCoJAyYMJkoEuD.png)

The **Qwen2-VL-OCR-2B-Instruct** model is a fine-tuned version of **Qwen/Qwen2-VL-2B-Instruct**, tailored for tasks that involve **Optical Character Recognition (OCR)**, **image-to-text conversion**, and **math problem solving with LaTeX formatting**. This model integrates a conversational approach with visual and textual understanding to handle multi-modal tasks effectively.

#### Key Enhancements:

* **SoTA understanding of images of various resolution & ratio**: Qwen2-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.

* **Understanding videos of 20min+**: Qwen2-VL can understand videos over 20 minutes for high-quality video-based question answering, dialog, content creation, etc.

* **Agent that can operate your mobiles, robots, etc.**: with the abilities of complex reasoning and decision making, Qwen2-VL can be integrated with devices like mobile phones, robots, etc., for automatic operation based on visual environment and text instructions.

* **Multilingual Support**: to serve global users, besides English and Chinese, Qwen2-VL now supports the understanding of texts in different languages inside images, including most European languages, Japanese, Korean, Arabic, Vietnamese, etc.

| **File Name**             | **Size**   | **Description**                                 | **Upload Status** |
|---------------------------|------------|------------------------------------------------|-------------------|
| `.gitattributes`          | 1.52 kB   | Configures LFS tracking for specific model files. | Initial commit    |
| `README.md`               | 203 Bytes | Minimal details about the uploaded model.       | Updated           |
| `added_tokens.json`       | 408 Bytes | Additional tokens used by the model tokenizer.  | Uploaded          |
| `chat_template.json`      | 1.05 kB   | Template for chat-based model input/output.     | Uploaded          |
| `config.json`             | 1.24 kB   | Model configuration metadata.                   | Uploaded          |
| `generation_config.json`  | 252 Bytes | Configuration for text generation settings.     | Uploaded          |
| `merges.txt`              | 1.82 MB   | BPE merge rules for tokenization.               | Uploaded          |
| `model.safetensors`       | 4.42 GB   | Serialized model weights in a secure format.    | Uploaded (LFS)    |
| `preprocessor_config.json`| 596 Bytes | Preprocessing configuration for input data.     | Uploaded          |
| `vocab.json`              | 2.78 MB   | Vocabulary file for tokenization.               | Uploaded          |

---
### Sample Inference with Doc

![123.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/TlsmcTqoQMvaBhwo8tGeU.png)

**📍Demo**: https://huggingface.co/prithivMLmods/Qwen2-VL-OCR-2B-Instruct/blob/main/Demo/ocrtest_qwen.ipynb
### How to Use

```python
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/Qwen2-VL-OCR-2B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
### Buf
```python
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        # Remove <|im_end|> or similar tokens from the output
        buffer = buffer.replace("<|im_end|>", "")
        yield buffer
```
### **Key Features**

1. **Vision-Language Integration:**  
   - Combines **image understanding** with **natural language processing** to convert images into text.  

2. **Optical Character Recognition (OCR):**  
   - Extracts and processes textual information from images with high accuracy.

3. **Math and LaTeX Support:**  
   - Solves math problems and outputs equations in **LaTeX format**.

4. **Conversational Capabilities:**  
   - Designed to handle **multi-turn interactions**, providing context-aware responses.

5. **Image-Text-to-Text Generation:**  
   - Inputs can include **images, text, or a combination**, and the model generates descriptive or problem-solving text.

6. **Secure Weight Format:**  
   - Uses **Safetensors** for faster and more secure model weight loading.

---

### **Training Details**

- **Base Model:** [Qwen/Qwen2-VL-2B-Instruct](#)  
- **Model Size:**  
   - 2.21 Billion parameters  
   - Optimized for **BF16** tensor type, enabling efficient inference.

- **Specializations:**  
   - OCR tasks in images containing text.
   - Mathematical reasoning and LaTeX output for equations.

---



Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
