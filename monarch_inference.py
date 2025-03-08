from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig

model_id = "AfricaComputeFund/Monarch-1"

# Explicitly load the correct model configuration (Mistral)
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config.model_type = "mistral"  # <-- Manually specifying the correct model type

# Load tokenizer and model explicitly with the adjusted config
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Who are you?"
response = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)

print(response[0]["generated_text"])
