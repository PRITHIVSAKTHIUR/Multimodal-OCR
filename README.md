# **Multimodal-OCR**

Multimodal-OCR is an experimental Gradio application for running OCR- and vision-oriented prompts against multiple pretrained vision-language models. The current interface accepts a single uploaded image, a single text instruction, and streams the selected model's raw text output.

The repository currently exposes six locally loaded models: `Nanonets-OCR2-3B`, `GLM-OCR`, `olmOCR-7B-0725`, `RolmOCR-7B`, `Aya-Vision-8B`, and `Qwen2-VL-OCR-2B`. It also exposes `DeepSeek-OCR` as an optional remote model backed by a compatible OpenAI-style endpoint.

<img width="1920" height="1800" alt="Screenshot 2026-03-22 at 12-44-23 Multimodal OCR - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/42b8c8f0-6903-4a83-96b8-04553fcc1df2" />

### **Key Features**

* **Seven Selectable Models:** Switch between six locally loaded models and one optional remote `DeepSeek-OCR` endpoint directly from the interface.
* **Single-Image Workflow:** The UI accepts one uploaded image and one text instruction per run.
* **Custom User Interface:** The app uses a custom Gradio frontend with drag-and-drop image upload, quick examples, advanced settings, and live output updates.
* **Generation Controls:** The interface exposes controls for Maximum New Tokens, Temperature, Top-p, Top-k, Repetition Penalty, and GPU Duration.
* **Raw Output Actions:** The visible result area streams raw text output and supports copy-to-clipboard and save-to-file actions.
* **Configured Model Loading:** The local model set is loaded through Hugging Face Transformers. `GLM-OCR` and `Aya-Vision-8B` are loaded through the standard image-text model API, while `DeepSeek-OCR` is called through a configured OpenAI-compatible endpoint.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   └── 5.jpg
├── app.py
├── LICENSE
├── pre-requirements.txt
├── README.md
└── requirements.txt
```

### **Installation and Requirements**

To run Multimodal-OCR locally, install the dependencies declared in this repository. The code selects `cuda:0` when CUDA is available and otherwise selects CPU. All local model loads request `attn_implementation="kernels-community/flash-attn2"` when CUDA is available, so runtime compatibility should be verified on the target environment.

**1. Install Pre-requirements**
Install the prerequisite declared in `pre-requirements.txt`:
```bash
python -m pip install -r pre-requirements.txt
```

**2. Install Core Requirements**
Install the main Python dependencies from `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

```text
git+https://github.com/huggingface/transformers.git@v4.57.6
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/peft.git
transformers-stream-generator
huggingface_hub
qwen-vl-utils
sentencepiece
opencv-python
torch==2.8.0
torchvision
matplotlib
requests
kernels
hf_xet
spaces
pillow
gradio
av
```

---

### **Usage**

Once your environment is set up and the dependencies are installed, launch the application with:

```bash
python app.py
```

At startup, the script launches the interface and loads local models lazily on first use. By default, the local runtime keeps one local model in memory at a time.

You can increase the warm local-model cache if you want faster switching and have enough VRAM for multiple resident models:

```bash
set LOCAL_MODEL_CACHE_SIZE=2
```

Higher cache sizes trade more VRAM for fewer reloads when switching between local models.

`DeepSeek-OCR` is not loaded in-process; instead, it becomes usable when `DEEPSEEK_OCR_BASE_URL` points to a compatible OpenAI-style endpoint. Until that endpoint is configured, the `DeepSeek-OCR` option stays visibly unavailable in the UI.

To enable `DeepSeek-OCR`, run the official vLLM server recommended by DeepSeek and set:

```bash
set DEEPSEEK_OCR_BASE_URL=http://127.0.0.1:8000/v1
set DEEPSEEK_OCR_API_KEY=EMPTY
```

An example official server command is:

```bash
vllm serve deepseek-ai/DeepSeek-OCR --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor --no-enable-prefix-caching --mm-processor-cache-gb 0
```

The current supported workflow is:

1. Upload one image through the web UI.
2. Enter one OCR or vision instruction.
3. Select one of the available model options.
4. Run inference and review the raw output stream.
5. Optionally copy the output or save it as a `.txt` file.

Bundled examples are available in the `examples/` directory and can be loaded from the Quick Examples section in the interface.

Current limitations visible from the code:

* The UI accepts one image at a time.
* Each run requires both an uploaded image and a text instruction.
* The visible output is raw streamed text from the selected model.

### **License and Source**

* **License:** Apache License - Version 2.0
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR.git](https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR.git)
