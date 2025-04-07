import torch

from transformers import AutoTokenizer, AutoModelForCausalLM



# 模型路径（以 Llama3.2-1B-Instruct 为例，你也可以试试 3B 或 8B）

model_path = "/share/apps/llama/Llama3.2-1B-Instruct"



# 加载 tokenizer 和模型

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")



# 输入 prompt

prompt = "用一句话解释什么是大语言模型（LLM）"



inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)



print("🦙 LLaMA 回复：\n")

print(response)


