import torch

model = AutoModelForCausalLM.from_pretrained("./finetuned-model")
tokenizer = AutoTokenizer.from_pretrained("./finetuned-model")

model.eval()
input_text = "晚餐要吃甚麼呢"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

with torch.no_grad():
    output = model.generate(input_ids, max_length=50)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)