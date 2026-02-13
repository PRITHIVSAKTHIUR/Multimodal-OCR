# **Multimodal-OCR**

> [!note]
HF Demo: https://huggingface.co/spaces/prithivMLmods/Multimodal-OCR

> A comprehensive multimodal OCR application that leverages state-of-the-art vision-language models for optical character recognition, document analysis, image captioning, and video understanding. This application provides a unified interface for multiple specialized models optimized for different OCR and vision tasks.

<img width="1918" height="1669" alt="Screenshot 2026-02-13 at 15-49-12 Multimodal OCR - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/00abc04c-4bb0-480a-a820-11b561892846" />

> [!important] 
note: remove kernels and flash_attn3 implementation if you are using it on *non-hopper* architecture gpus.

## Features

### Core Capabilities
- **Advanced OCR**: Extract text from images with high accuracy using multiple specialized models
- **Document Analysis**: Convert documents to structured formats including markdown and tables
- **Image Understanding**: Comprehensive scene analysis and visual reasoning
- **Video Processing**: Temporal video analysis with frame-by-frame understanding
- **Multi-Model Support**: Choose from five different vision-language models optimized for specific tasks

### Supported Models
1. **Nanonets-OCR-s**: State-of-the-art image-to-markdown OCR with intelligent content recognition
2. **Qwen2-VL-OCR-2B**: Specialized for messy OCR, image-to-text conversion, and math problem solving
3. **RolmOCR-7B**: High-quality document parsing for PDFs and complex layouts
4. **olmOCR-7B-0725**: fine-tuned with olmocr-mix-0225 on top of Qwen2.5-VL-7B-Instruct
5. **Aya-Vision-8B**: Multi-purpose vision-language model with advanced reasoning capabilities

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR.git
cd Multimodal-OCR

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies
```
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
gradio # - gradio@6.3.0
av
```

## Usage

### Running the Application
```bash
python app.py
```

The application will launch a Gradio web interface accessible through your browser.

### Interface Overview

#### Image Inference Tab
- **Upload Images**: Support for various image formats (PNG, JPG, etc.)
- **Query Input**: Natural language queries for specific tasks
- **Model Selection**: Choose from five specialized models
- **Real-time Processing**: Streaming responses with live output

### Advanced Configuration Options
- **Max New Tokens**: Control response length (1-2048)
- **Temperature**: Adjust creativity and randomness (0.1-4.0)
- **Top-p**: Nuclear sampling parameter (0.05-1.0)
- **Top-k**: Token selection threshold (1-1000)
- **Repetition Penalty**: Prevent repetitive outputs (1.0-2.0)

## API Reference

### Core Functions

#### `generate_image(model_name, text, image, **kwargs)`
Process image inputs with the selected model
- **Parameters**: 
  - `model_name`: Selected model identifier
  - `text`: Query or instruction text
  - `image`: PIL Image object
  - `**kwargs`: Generation parameters (temperature, top_p, etc.)
- **Returns**: Streaming tuple of (raw_output, markdown_output)

### Utility Functions

#### `downsample_video(video_path)`
Extract representative frames from video files
- **Parameters**: Path to video file
- **Returns**: List of (PIL_image, timestamp) tuples
- **Frame Sampling**: Extracts 10 evenly spaced frames

## Examples

### Document OCR
```python
# Query: "Convert this page to doc [table] precisely for markdown"
# Input: Document image with tables
# Output: Structured markdown with proper table formatting
```

### Scene Analysis
```python
# Query: "Explain the scene"
# Input: Complex image
# Output: Detailed scene description with context
```

### Mathematical Content
```python
# Query: "Extract the mathematical equations"
# Input: Image with equations
# Output: LaTeX formatted mathematical expressions
```

## Performance Optimization

### GPU Acceleration
- CUDA support for all models
- Automatic device detection
- Memory-efficient model loading

### Streaming Responses
- Real-time output generation
- Progressive result display
- Reduced perceived latency

## Configuration

### Environment Variables
```bash
MAX_INPUT_TOKEN_LENGTH=4096  # Maximum input token length
```

### Model Parameters
- **Default Max Tokens**: 1024
- **Default Temperature**: 0.6
- **Default Top-p**: 0.9
- **Default Top-k**: 50
- **Default Repetition Penalty**: 1.2

## Limitations

- GPU memory requirements scale with model complexity
- Processing time depends on input size and hardware capabilities
- Some models may not perform optimally on video content

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development

### Local Development
```bash
# Install in development mode
pip install -e .

# Run with debugging
python app.py --debug
```

### Testing
```bash
# Run basic functionality tests
python -m pytest tests/

# Test individual models
python test_models.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Built on Hugging Face Transformers ecosystem
- Powered by Gradio for interactive web interface
- Utilizes multiple state-of-the-art vision-language models
- Integrated with Spaces GPU acceleration platform

## Support and Issues

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the model documentation on Hugging Face
- Join the discussion in the Hugging Face Space
