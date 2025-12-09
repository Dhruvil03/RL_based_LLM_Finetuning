from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("checkpoints/final")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/final")

prompt = "What is 12 - 7? Answer with just the integer."
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=3,
    do_sample=False,   # <- deterministic, not random
    temperature=1.0,
    top_p=1.0
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
