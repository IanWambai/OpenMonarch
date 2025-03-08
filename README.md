# Monarch: African-Optimized AI Model

## Overview

Monarch is a project built around the Monarch-1 model, a generative AI system fine-tuned from Mistral-7B-Instruct-v0.3 and optimized specifically for African linguistic, cultural, and economic contexts. Developed within the Africa Compute Fund (ACF), this project aims to bridge the gap between global AI models and Africa's unique needs.

## Features

- **African Context Optimization**: Enhanced understanding of diverse languages, historical contexts, and market-specific data across the African continent.
- **Efficient Inference**: Optimized for both cloud and edge deployment with proper resource utilization.
- **API Integration**: Simple FastAPI implementation for easy integration with applications.
- **Docker Support**: Containerized deployment for consistent performance across environments.
- **Fine-tuning Tools**: Includes scripts for LoRA (Low-Rank Adaptation) fine-tuning and model merging.

## Technical Stack

- Base Model: [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- Fine-tuning Method: Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- Training Framework: AutoTrain by Hugging Face
- Inference: Transformers library, optimized with PyTorch
- Deployment: Docker, FastAPI

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for inference)
- 16GB+ RAM

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/monarch.git
cd monarch

# Create and activate virtual environment
python -m venv monarch_env
source monarch_env/bin/activate  # On Windows: monarch_env\Scripts\activate

# Install dependencies
pip install torch transformers fastapi uvicorn peft
```

## Usage

### Model Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local inference
tokenizer = AutoTokenizer.from_pretrained("AfricaComputeFund/Monarch-1")
model = AutoModelForCausalLM.from_pretrained(
    "AfricaComputeFund/Monarch-1",
    device_map="auto",
    torch_dtype="auto"
)

# Example prompt
messages = [
    {"role": "user", "content": "What impact can Monarch-1 have in Africa?"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

print(response)
```

### API Deployment

```bash
# Run API locally
uvicorn monarch_api:app --host 0.0.0.0 --port 8000

# Using Docker
docker build -t monarch:latest .
docker run -p 8000:8000 monarch:latest
```

## Model Merging

If you've fine-tuned your own LoRA adapter and want to merge it back with the base model:

```bash
python merge_lora.py
```

You can modify the `BASE_MODEL_ID`, `LORA_ADAPTER_PATH`, and `OUTPUT_DIR` variables in the script according to your needs.

## Project Structure

```
monarch/
├── README.md                 # This file
├── monarch_inference.py      # Basic inference script
├── monarch_api.py            # FastAPI implementation
├── merge_lora.py             # LoRA merging utility
├── Dockerfile                # Container configuration
├── extract_pdf.py            # Data extraction utility
├── extract_masakhaner.py     # NER dataset extraction
├── clean_data.py             # Data preprocessing
├── The Monarch Benchmark.pdf # Evaluation documentation
└── dataset/                  # Training data directory
```

## Ethical Use and Responsibility

Monarch-1 is designed for ethical and responsible AI use. When using this model:

- Avoid generating harmful, biased, or misleading content
- Ensure culturally sensitive responses
- Use the model in applications that align with constructive, transparent, and ethical AI deployment

## License

This project uses the Monarch-1 model which has its own license. See [the model card](https://huggingface.co/AfricaComputeFund/Monarch-1) for more details.

## Acknowledgements

This project leverages the Monarch-1 model from the Africa Compute Fund (ACF), fine-tuned from Mistral AI's Mistral-7B-Instruct-v0.3. 