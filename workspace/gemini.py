import sys, os, json, re
import litellm
import base64
import asyncio

os.environ["LITELLM_PROXY_API_BASE"] = "http://8.219.58.57:4000"
os.environ["LITELLM_PROXY_API_KEY"] = "sk-WNrS8wC5RXbYvAx6KKdyEw"

def eval(image_path, prompt):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                    }
                ]
            }
        ]
        response = litellm.completion(
            model="litellm_proxy/gemini-3-pro-preview",
            messages=messages,
            max_tokens=2048
        )
        response = response['choices'][0]['message']['content']
        return response
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

image_path = "./output/has_reason/826738/826738_2.png"
prompt = "**交替共聚物 (Alternating copolymer)**：红蓝单体严格交替排列。请问图片中的内容符合对于交替聚合物的描述吗？"
print(eval(image_path, prompt))