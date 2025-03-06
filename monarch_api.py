from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

MODEL_PATH = "/app/deepseek-llm-8b-chat"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

@app.post("/")
async def generate_text(request: dict):
    try:
        input_text = request.get("input", "")
        if not input_text:
            raise HTTPException(status_code=400, detail="Input text is required")

        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        output_ids = model.generate(**inputs, max_length=200)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {"output": output_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
