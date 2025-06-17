# **Multimodal OCR**

A comprehensive optical character recognition (OCR) application that supports both image and video inputs using multiple state-of-the-art vision-language models.

## Features

- **Multi-Model Support**: Choose from 4 different OCR models optimized for various use cases
- **Image OCR**: Extract text and analyze content from images
- **Video OCR**: Process video files with frame-by-frame analysis
- **Interactive Web Interface**: User-friendly Gradio interface with real-time streaming
- **Advanced Configuration**: Customizable generation parameters for optimal results

## Supported Models

### Nanonets-OCR-s
A powerful image-to-markdown OCR model that transforms documents into structured markdown with intelligent content recognition and semantic tagging.

### Qwen2-VL-OCR-2B-Instruct
Fine-tuned version of Qwen2-VL-2B-Instruct, specialized for messy optical character recognition, image-to-text conversion, and math problem solving with LaTeX formatting.

### RolmOCR
High-quality approach to parsing PDFs and complex documents. Designed to handle scanned documents, handwritten text, and complex layouts.

### Aya-Vision
8-billion parameter model with advanced capabilities for vision-language tasks including OCR, captioning, visual reasoning, summarization, and question answering.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR.git
cd Multimodal-OCR
```

2. Install required dependencies:
```bash
pip install torch torchvision torchaudio
pip install transformers
pip install gradio
pip install spaces
pip install opencv-python
pip install pillow
pip install numpy
```

## Usage

### Running the Application

```bash
python app.py
```

The application will launch a Gradio interface accessible through your web browser.

### Image OCR

1. Navigate to the "Image Inference" tab
2. Enter your query (e.g., "Perform OCR on the Image" or "Extract table content")
3. Upload an image file
4. Select your preferred model
5. Click "Submit" to process

### Video OCR

1. Navigate to the "Video Inference" tab
2. Enter your query (e.g., "Explain the Ad in Detail")
3. Upload a video file
4. Select your preferred model
5. Click "Submit" to process

The application will automatically extract 10 evenly spaced frames from the video for analysis.

## Configuration Options

### Advanced Parameters

- **Max New Tokens**: Control the maximum length of generated responses (1-2048)
- **Temperature**: Adjust randomness in text generation (0.1-4.0)
- **Top-p**: Configure nucleus sampling (0.05-1.0)
- **Top-k**: Set the number of highest probability tokens to consider (1-1000)
- **Repetition Penalty**: Prevent repetitive text generation (1.0-2.0)

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for optimal performance
- **RAM**: Minimum 8GB, 16GB+ recommended for larger models
- **Storage**: At least 20GB free space for model downloads

## Model Performance

Each model has specific strengths:

- **Nanonets-OCR-s**: Best for structured document parsing and markdown conversion
- **Qwen2-VL-OCR-2B-Instruct**: Excellent for mathematical content and messy text
- **RolmOCR**: Optimal for complex layouts and handwritten text
- **Aya-Vision**: Superior for general vision-language tasks and reasoning

## Examples

### Image Processing Examples
- "Perform OCR on the Image" - Extract all text from document images
- "Extract the table content" - Parse tabular data from images

### Video Processing Examples
- "Explain the Ad in Detail" - Analyze advertisement content across video frames
- "Identify the main actions in the cartoon video" - Describe actions and events in animated content

## Technical Details

### Video Processing
The application downsamples videos to 10 evenly distributed frames, maintaining temporal information with timestamps for comprehensive analysis.

### Model Loading
All models are loaded with half-precision (float16) for memory efficiency and use GPU acceleration when available.

### Streaming Output
Real-time text generation with streaming capabilities for immediate feedback during processing.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source. Please refer to the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- Gradio for the web interface
- Individual model creators and maintainers
- OpenCV for video processing capabilities

## Repository

GitHub: https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR.git
