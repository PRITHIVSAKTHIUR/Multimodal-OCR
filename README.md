# **Multimodal-OCR**

Multimodal-OCR is an experimental, high-performance visual reasoning and optical character recognition suite designed to accurately extract text, analyze visual content, and parse complex document structures. Built upon a diverse ecosystem of cutting-edge vision-language models—including architectures based on Qwen2.5-VL, Qwen2-VL, and Cohere's Aya-Vision—this application excels at handling dense documents, multilingual texts, and real-world scene imagery. The suite features a highly customized, interactive web interface that enables users to effortlessly upload screenshots, receipts, and pages for rapid analysis. With built-in support for fully GPU-accelerated inference via Flash Attention 2 and granular manipulation of text generation parameters, Multimodal-OCR provides researchers and developers with a powerful, streamlined environment for testing and deploying robust multimodal AI workflows.

<img width="1920" height="1800" alt="Screenshot 2026-03-22 at 12-44-23 Multimodal OCR - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/42b8c8f0-6903-4a83-96b8-04553fcc1df2" />

### **Key Features**

* **Multi-Model Architecture:** Seamlessly switch between specialized vision-language models directly from the interface. Supported models include `Nanonets-OCR2-3B`, `olmOCR-7B-0725`, `RolmOCR-7B`, `Aya-Vision-8B`, and `Qwen2-VL-OCR-2B`.
* **Custom User Interface:** Features a bespoke, responsive Gradio frontend built with custom HTML, CSS, and JavaScript. It includes a drag-and-drop media zone, real-time output streaming, and an integrated advanced settings panel.
* **Granular Inference Controls:** Fine-tune the AI's output by adjusting text generation parameters such as Maximum New Tokens, Temperature, Top-p, Top-k, and Repetition Penalty.
* **Output Management:** Built-in actions allow users to instantly copy the raw output text to their clipboard or save the generated response directly as a `.txt` file.
* **Flash Attention 2 Integration:** Utilizes `kernels-community/flash-attn2` for optimized, memory-efficient inference on compatible GPUs.

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

To run Multimodal-OCR locally, you need to configure a Python environment with the following dependencies. Ensure you have a compatible CUDA-enabled GPU for optimal performance.

**1. Install Pre-requirements**
Run the following command to update pip to the required version:
```bash
pip install pip>=23.0.0
```

**2. Install Core Requirements**
Install the necessary machine learning and UI libraries. You can place these in a `requirements.txt` file and run `pip install -r requirements.txt`.

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

Once your environment is set up and the dependencies are installed, you can launch the application by running the main Python script:

```bash
python app.py
```

After the script initializes the interface, it will provide a local web address (usually `http://127.0.0.1:7860/`) which you can open in your browser to interact with the models. Note that the selected models will be downloaded and loaded into VRAM upon their first invocation.

### **License and Source**

* **License:** Apache License - Version 2.0
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR.git](https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR.git)
