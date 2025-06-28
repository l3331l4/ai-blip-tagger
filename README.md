# AI Tagger

A tool for automatically generating descriptive captions for images and videos using the BLIP model.

## Features

- Supports image (JPG, PNG, WEBP, BMP, TIFF) and video (MP4, AVI, MOV, MKV, GIF) formats
- Automatically skip files that have already been tagged
- Exports to CSV with optional metadata
- GPU acceleration via CUDA if available
- Fast tokenizer mode for processing large batches

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process a single image
python main.py test_image.jpg

# Process a directory with detailed output
python main.py photos/ --batch --v

# Custom output format example
python main.py videos/ -o results.csv --format basic
```

## Installation

```bash
git clone https://github.com/l3331l4/ai-tagger.git
cd ai-tagger
pip install -r requirements.txt
```

## Output Formats

### Detailed (Default)
```csv
filename,caption,size_kb,dimensions,file_type,date_processed
```

### Basic
```csv
filename,caption
```

## Why This Project?

I created this while working with AI-generated images and videos and found it tedious to manually organize the massive number of outputs. This tool helps automate that and is especially useful when dealing with big batches of SD/AnimateDiff content.

Helpful for preparing rough CSV datasets or just organizing large batches of AI outputs by content.

![Chaos to order metaphor](assets/chaos_to_order.png)

## Requirements

- Python 3.8+
- PyTorch 2.6+
- Transformers 4.20+
- CUDA (optional, for GPU acceleration)