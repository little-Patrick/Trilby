<h1 align="center"> Trilby - Code Completion Model Training Pipeline </h1>

A comprehensive pipeline for training code completion models using LoRA fine-tuning on The Stack v2 dataset with Fill-in-the-Middle (FIM) capabilities.

## Overview

Trilby is a modular training pipeline designed to fine-tune small large language model adapters for code completion tasks. It features:

- **Fill-in-the-Middle (FIM) Training**: Enables models to complete code in the middle of existing context
- **LoRA Fine-tuning**: Memory-efficient training using Low-Rank Adaptation
- **Quantized Training**: 4-bit and 8-bit quantization support for training on consumer GPUs
- **Modular Dataset Pipeline**: Easy integration with different code datasets
- **Interactive Testing**: Command-line runner for testing trained models

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (6GB+ VRAM recommended)
- AWS credentials (If youw want to use bigcode's The Stack v2 dataset)

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

##  Quick Start

### 1. Configure Training

    - Copy the configuration file at `training_configs/template.json`.
    - Configure and tune parameters 


### 2. Train the Model

```bash
python TrilbyTrainer.py
```

### 3. Test the Model

```bash
python model_runner.py
```

## Hardware Requirements

| GPU Memory | Recommended Settings |
|------------|---------------------|
| 6GB (GTX 1060) | batch_size=1, 4-bit quantization |
| 8GB (RTX 3070) | batch_size=2, 4-bit quantization |
| 12GB+ (RTX 3080+) | batch_size=4+, 8-bit quantization |

## Training Features

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

## Pipeline Roadmap
- [x] Basic training pipeline with LoRA
- [ ] Fine tune FIM Tokenization
- [ ] Fine tune data preparation
    - [ ] Test different data sources
    - [ ] Better data filtering
    - [ ] Fim ratio
- [ ] Expand pipeline to use different base models
    - Starting Base Models:
        - [x] Llama
        - [ ] Phi
        - [ ] Qwen
        - [ ] DeepSeek R1
        - [ ] NanoGPT and DistilGPT

## Training Roadmap

### Local Training
- [x] Train locally to test pipeline
- [ ] Train multiple language adapters
    - Starting with:
    - [x] Python
    - [ ] Javascript
    - [ ] Go
    - [ ] Bash/Zsh
    - [ ] SQL

## Cloud Training
- [ ] Create Docker container
- [ ] Fine tune cloud config files

