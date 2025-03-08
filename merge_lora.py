import os
import shutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration:
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # The base model used for fine-tuning
LORA_ADAPTER_PATH = "Monarch-1"  # Your adapter repo folder (contains adapter_model.safetensors)
OUTPUT_DIR = "merged-monarch"    # Directory where the merged model will be saved

def get_model_size_bytes(model):
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    return total_bytes

def bytes_to_gb(b):
    return b / (1024**3)

def main():
    print("="*80)
    print("Step 1: Loading base model from:", BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16  # Use torch.float32 if you're on CPU
    )
    print("Base model loaded successfully.")
    print("="*80)
    
    print("Step 2: Loading LoRA adapter from:", LORA_ADAPTER_PATH)
    lora_model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_PATH,
        torch_dtype=torch.float16
    )
    print("LoRA adapter loaded successfully.")
    print("="*80)
    
    print("Step 3: Merging LoRA adapter weights into the base model...")
    merged_model = lora_model.merge_and_unload()
    print("Merging complete.")
    print("="*80)
    
    # Estimate model size
    estimated_size = get_model_size_bytes(merged_model)
    print("Estimated merged model size: {:.2f} GB".format(bytes_to_gb(estimated_size)))
    
    # Check available disk space in the parent directory of OUTPUT_DIR
    parent_dir = os.path.abspath(os.path.join(OUTPUT_DIR, os.pardir))
    usage = shutil.disk_usage(parent_dir)
    free_space = usage.free
    print("Free disk space available: {:.2f} GB".format(bytes_to_gb(free_space)))
    
    if free_space < estimated_size:
        print("WARNING: Not enough disk space to save the merged model.")
        print("You need approximately {:.2f} GB free, but only have {:.2f} GB available.".format(bytes_to_gb(estimated_size), bytes_to_gb(free_space)))
        return
    else:
        print("Sufficient disk space available. Proceeding with saving the model...")
    print("="*80)
    
    print("Step 4: Saving the merged model to", OUTPUT_DIR, "...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_DIR)
    print("Merged model saved successfully in", OUTPUT_DIR)
    print("="*80)
    
    print("Step 5: Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Tokenizer saved successfully in", OUTPUT_DIR)
    print("="*80)
    
    print("Merge complete. The merged model is saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
