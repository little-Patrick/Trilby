# Trilby - Code Completion Model Training Pipeline

A comprehensive pipeline for training code completion models using LoRA fine-tuning on The Stack v2 dataset with Fill-in-the-Middle (FIM) capabilities.

## Overview

Trilby is a modular training pipeline designed to fine-tune large language models for code completion tasks. It features:

- **Fill-in-the-Middle (FIM) Training**: Enables models to complete code in the middle of existing context
- **LoRA Fine-tuning**: Memory-efficient training using Low-Rank Adaptation
- **Quantized Training**: 4-bit and 8-bit quantization support for training on consumer GPUs
- **Modular Dataset Pipeline**: Easy integration with different code datasets
- **Interactive Testing**: Command-line runner for testing trained models

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (6GB+ VRAM recommended)
- AWS credentials (for The Stack v2 dataset access)

### Dependencies

```bash
pip install torch transformers datasets peft bitsandbytes
pip install smart-open boto3 accelerate
```

### AWS Setup

Set your AWS credentials for accessing The Stack v2 dataset:

```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
```

## 🚀 Quick Start

### 1. Configure Training

Edit the configuration file at `training_configs/template.json`:

### 2. Train the Model

```bash
python TrilbyTrainer.py
```

### 3. Test the Model

```bash
python model_runner.py
```

## 💻 Hardware Requirements

| GPU Memory | Recommended Settings |
|------------|---------------------|
| 6GB (GTX 1060) | batch_size=1, 4-bit quantization |
| 8GB (RTX 3070) | batch_size=2, 4-bit quantization |
| 12GB+ (RTX 3080+) | batch_size=4+, 8-bit quantization |

## 📊 Training Features

### Fill-in-the-Middle (FIM)
- Trains models to complete code in the middle of existing context
- Uses special tokens: `<fim_prefix>`, `<fim_middle>`, `<fim_suffix>`
- Configurable FIM ratio in training data

### LoRA Configuration
- Low-rank adaptation for memory-efficient fine-tuning
- Targets attention layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Configurable rank (r) and alpha parameters

### Quantization Support
- 4-bit and 8-bit quantization using BitsAndBytes
- Enables training large models on consumer hardware
- Gradient checkpointing for further memory optimization

## 🎯 Usage Examples

### Normal Code Completion
```python
# Input
def calculate_fibonacci(n):

# Generated Output
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
```

### Fill-in-the-Middle
```python
# Prefix
def process_data(data):
    cleaned_data = clean(data)

# Suffix
    return result

# Generated Middle
    processed = transform(cleaned_data)
    result = validate(processed)
```

## 📈 Roadmap

### Phase 1: Core Pipeline ✅
- [x] Basic training pipeline with LoRA
- [x] FIM token integration
- [x] The Stack v2 dataset support
- [x] Quantized training (4-bit/8-bit)
- [x] Interactive model runner

### Phase 2: Enhanced Training
- [ ] Multi-language support (JavaScript, Java, Go, etc.)
- [ ] Advanced data filtering and preprocessing
- [ ] Sequence packing for better GPU utilization
- [ ] Mixed precision training optimization
- [ ] Distributed training support

### Phase 3: Evaluation & Metrics
- [ ] Code completion benchmarks (HumanEval, MBPP)
- [ ] FIM-specific evaluation metrics
- [ ] Perplexity and accuracy tracking
- [ ] A/B testing framework for model comparison

### Phase 4: Production Features
- [ ] Model serving API (FastAPI/Flask)
- [ ] VSCode extension integration
- [ ] Batch inference optimization
- [ ] Model quantization for deployment
- [ ] Docker containerization

### Phase 5: Advanced Capabilities
- [ ] Multi-modal code completion (code + comments)
- [ ] Repository-level context awareness
- [ ] Code explanation and documentation generation
- [ ] Vulnerability detection integration
- [ ] Custom tokenizer training

### Phase 6: Research & Innovation
- [ ] Novel architecture experiments
- [ ] Retrieval-augmented code completion
- [ ] Code style adaptation
- [ ] Cross-language code translation
- [ ] Integration with code analysis tools


- **The Stack v2**: BigCode for providing the training dataset
- **LoRA**: Microsoft for the Low-Rank Adaptation technique
- **BitsAndBytes**: For enabling quantized training
- **Hugging Face**: For the transformers libr
- **LLM-Workshop**: [Github Link](https://github.com/pacman100/LLM-Workshop/blob/main/personal_copilot/training/train.py)