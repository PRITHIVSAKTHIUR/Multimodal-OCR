# Multimodal-OCR

<p align="center">
	<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python badge" />
	<img src="https://img.shields.io/badge/Gradio-UI-F97316?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio badge" />
	<img src="https://img.shields.io/badge/PyTorch-Inference-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch badge" />
	<img src="https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Transformers badge" />
	<img src="https://img.shields.io/badge/DeepSeek-Remote%20OCR-1E40AF?style=for-the-badge" alt="DeepSeek badge" />
	<img src="https://img.shields.io/badge/License-Apache%202.0-2EA44F?style=for-the-badge" alt="License badge" />
</p>

<p align="center"><strong>📄 OCR + 👁️ vision prompting across local and remote multimodal models</strong></p>

Multimodal-OCR is a Gradio application for OCR and vision-oriented prompting across multiple vision-language models. The app accepts a single uploaded image and a single text instruction, runs the selected backend, and streams the generated text output back into the UI.

It currently supports six local models and one optional remote DeepSeek-OCR endpoint.

<img width="1920" height="1800" alt="Multimodal-OCR application screenshot" src="https://github.com/user-attachments/assets/42b8c8f0-6903-4a83-96b8-04553fcc1df2" />

## ✨ Highlights

- Seven selectable backends: six local models plus one optional remote DeepSeek-OCR endpoint.
- Lazy local model loading with a bounded in-memory cache controlled by `LOCAL_MODEL_CACHE_SIZE`.
- Streaming text output and backend-driven run-status updates in the UI.
- Custom Gradio interface with drag-and-drop upload, quick examples, advanced generation controls, and output actions.
- Built-in environment loading through `.env` via `python-dotenv`.
- Explicit CPU startup warning when CUDA is unavailable.

## 🧠 Supported Models

| Model | Execution Mode | Backend Type | Notes |
| --- | --- | --- | --- |
| Nanonets-OCR2-3B | Local | `Qwen2_5_VLForConditionalGeneration` | OCR-focused local model |
| GLM-OCR | Local | `AutoModelForImageTextToText` | Uses `Text Recognition:` as the default prompt |
| olmOCR-7B-0725 | Local | `Qwen2_5_VLForConditionalGeneration` | OCR-focused local model |
| RolmOCR-7B | Local | `Qwen2_5_VLForConditionalGeneration` | OCR-focused local model |
| Aya-Vision-8B | Local | `AutoModelForImageTextToText` | General vision-language model |
| Qwen2-VL-OCR-2B | Local | `Qwen2VLForConditionalGeneration` | OCR-focused local model |
| DeepSeek-OCR | Remote | OpenAI-compatible `/chat/completions` API | Disabled in the UI until `DEEPSEEK_OCR_BASE_URL` is set |

## 🛠️ Tech Stack

<p>
	<img src="https://skillicons.dev/icons?i=python,pytorch" alt="Python and PyTorch logos" />
	<img src="https://go-skill-icons.vercel.app/api/icons?i=gradio" alt="Gradio logo" height="48" />
	<img src="https://go-skill-icons.vercel.app/api/icons?i=huggingface" alt="Hugging Face logo" height="48" />
</p>

- 🐍 Python application with a Gradio-based custom UI.
- 🔥 PyTorch-backed local inference for the in-process models.
- 🤗 Transformers-powered processor and model loading.
- 🌐 Requests-based OpenAI-compatible remote integration for DeepSeek-OCR.
- ⚙️ `.env`-driven configuration using `python-dotenv`.

## ⚙️ Runtime Behavior

### 🖥️ Local Models

- Local models are loaded on demand rather than eagerly at startup.
- Loaded local models are stored in an `OrderedDict`-backed LRU cache.
- Cache size is controlled by `LOCAL_MODEL_CACHE_SIZE` and defaults to `1`.
- When CUDA is unavailable, the app removes `attn_implementation` from local model load kwargs and upgrades `torch.float16` model loads to `torch.float32`.
- The app emits a startup warning when running without CUDA because CPU inference may be extremely slow or fail for larger models.

### ☁️ Remote DeepSeek-OCR

- DeepSeek-OCR is not loaded in-process.
- Requests are sent to an OpenAI-compatible endpoint at `DEEPSEEK_OCR_BASE_URL/chat/completions`.
- The DeepSeek tab remains visible but unavailable until `DEEPSEEK_OCR_BASE_URL` is configured.

### ✅ Validation and Output

- Each run requires exactly one uploaded image and one text instruction.
- Prompt length is validated before inference using `MAX_INPUT_TOKEN_LENGTH * 4` for both local and DeepSeek paths.
- Output is streamed as raw text and can be copied or saved from the UI.

## 🚀 Quick Start

### 1. Install Prerequisites

```bash
python -m pip install -r pre-requirements.txt
```

### 2. Install Runtime Dependencies

```bash
python -m pip install -r requirements.txt
```

The Python dependencies are maintained in `requirements.txt`. Major runtime components include Gradio, PyTorch, Transformers-based model loaders, Requests, and `python-dotenv`.

### 3. Configure Environment Variables

The app automatically loads variables from a local `.env` file if present. You can either copy `.env.example` or set variables directly in your shell.

Example `.env`:

```text
LOCAL_MODEL_CACHE_SIZE=2
MAX_INPUT_TOKEN_LENGTH=4096
DEEPSEEK_OCR_BASE_URL=http://127.0.0.1:8000/v1
DEEPSEEK_OCR_API_KEY=EMPTY
DEEPSEEK_OCR_MODEL_ID=deepseek-ai/DeepSeek-OCR
DEEPSEEK_OCR_TIMEOUT=3600
LOG_LEVEL=INFO
```

### 4. Start the App

```bash
python app.py
```

## 🔧 Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOCAL_MODEL_CACHE_SIZE` | `1` | Maximum number of resident local models kept in the LRU cache |
| `MAX_INPUT_TOKEN_LENGTH` | `4096` | Base input-length control used by local preprocessing and prompt validation |
| `DEEPSEEK_OCR_BASE_URL` | empty | Enables the DeepSeek-OCR remote tab when set |
| `DEEPSEEK_OCR_API_KEY` | `EMPTY` | Authorization value for the OpenAI-compatible DeepSeek endpoint |
| `DEEPSEEK_OCR_MODEL_ID` | `deepseek-ai/DeepSeek-OCR` | Model name sent in remote DeepSeek requests |
| `DEEPSEEK_OCR_TIMEOUT` | `3600` | Request timeout in seconds for DeepSeek remote calls |
| `LOG_LEVEL` | `INFO` | Application logging level |

## 🌐 DeepSeek-OCR Setup

To enable DeepSeek-OCR, start a compatible server and point `DEEPSEEK_OCR_BASE_URL` at it.

Example environment values:

```text
DEEPSEEK_OCR_BASE_URL=http://127.0.0.1:8000/v1
DEEPSEEK_OCR_API_KEY=EMPTY
```

Example `vllm` command:

```bash
vllm serve deepseek-ai/DeepSeek-OCR --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor --no-enable-prefix-caching --mm-processor-cache-gb 0
```

Once configured, the DeepSeek tab becomes selectable in the UI.

## 🧪 Using the App
 
1. 🚀 Launch the app with `python app.py`.
2. 🖼️ Upload one image through the drag-and-drop area or file picker.
3. 🧠 Select a model tab.
4. ✍️ Enter an OCR or vision instruction.
5. 🎛️ Adjust generation controls if needed.
6. ▶️ Run inference and watch the streamed text output.
7. 💾 Copy the result to the clipboard or save it as a `.txt` file.
 
Quick Examples in the UI populate the image, prompt, and model selection automatically using the bundled files in `examples/`.
 
## 📁 Repository Layout
 
```text
├── .env.example
├── examples/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   ├── 5.jpg
│   ├── 6.png
│   └── 7.png
├── app.py
├── LICENSE
├── pre-requirements.txt
├── README.md
└── requirements.txt
```
 
## 📌 Operational Notes
 
- ⚡ CUDA is strongly recommended for local inference.
- 🐢 CPU execution is supported by the current code path but may be too slow or memory-intensive for larger local models.
- 🖼️ The app currently supports one image per run.
- 📝 The visible output is raw streamed text from the selected backend.
- 🌍 DeepSeek-OCR requires an external server and is not available offline by default.
 
## 📄 License
 
This project is licensed under the Apache License 2.0. See `LICENSE` for details.
 
## 🔗 Source
 
GitHub repository: https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR.git
