import os

import json

import requests

from dotenv import load_dotenv



# 加载 .env 文件

load_dotenv()



# 读取环境变量

api_key = os.getenv("OPENROUTER_API_KEY")

referer = os.getenv("OPENROUTER_REFERER", "")

title = os.getenv("OPENROUTER_TITLE", "OpenRouterTest")



# 构造请求

headers = {

    "Authorization": f"Bearer {api_key}",

    "HTTP-Referer": referer,

    "X-Title": title,

    "Content-Type": "application/json"

}



payload = {

    "model": "openai/gpt-4o",

    "messages": [

        {"role": "user", "content": "用一句话解释什么是大型语言模型（LLM）"}

    ]

}



# 发送请求

response = requests.post(

    url="https://openrouter.ai/api/v1/chat/completions",

    headers=headers,

    data=json.dumps(payload)

)



# 解析并输出结果

if response.status_code == 200:

    result = response.json()

    print("✅ GPT-4o 回答：\n")

    print(result["choices"][0]["message"]["content"])

else:

    print("❌ 请求失败：")

    print(f"状态码：{response.status_code}")

    print(f"内容：{response.text}")


