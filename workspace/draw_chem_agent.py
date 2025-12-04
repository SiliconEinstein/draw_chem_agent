import sys, os, json, re
sys.path.append("F:/SciencePedia")
import litellm
import base64
import asyncio
import logging
import markdown
from typing import Dict, List

from organic_chem.get_article import fetch_article_content

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["LITELLM_PROXY_API_BASE"] = "http://8.219.58.57:4000"
os.environ["LITELLM_PROXY_API_KEY"] = "sk-WNrS8wC5RXbYvAx6KKdyEw"

class DrawChemAgent(object):
    def __init__(self):
        self.model_gemini_2_5_pro = "litellm_proxy/gemini-2.5-pro"
        self.model_gemini_3_pro = "litellm_proxy/gemini-3-pro-preview"
        self.model_gemini_3_pro_image = "litellm_proxy/gemini-3-pro-image-preview"    
        
        self.model_kwargs = {}

    def get_article(self, article_id):
        return fetch_article_content(article_id)
    
    def get_prompt(self, prompt_path: str, prompt_name: str, content: Dict[str, str]) -> str:
        """
        加载并填充 prompt 模板
        """
        file_path = f"{prompt_path}/{prompt_name}"
        with open(file_path, "r", encoding="utf-8") as f:
            prompt = f.read()
        for key, value in content.items():
            prompt = prompt.replace(key, value)
        return prompt
    
    def parse_result(self, response: str) -> str:
        match = re.search(r"```(?:json)?\s*([\s\S]*)\s*```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()
    
    # 插入标记符
    def insert_marker(self, article_id, positions, main_content, applications, output_dir=""):
        def match(pos, line):
            i,j = 0,0
            while i < len(pos) and j < len(line):
                if pos[i] == line[j]:
                    i+=1
                j+=1
            return i == len(pos)
        index = 0
        main_content_lines = []
        for i, line in enumerate(main_content.split("\n")):
            main_content_lines.append(line)
            if index >= len(positions):
                main_content_lines.extend(main_content.split("\n")[i:])
                break
            if match(positions[index], line):
                main_content_lines.append(f"\n(@images:{article_id}_{index}.png)")
                index += 1
            
        applications_lines = []
        for i, line in enumerate(applications.split("\n")):
            applications_lines.append(line)
            if index >= len(positions):
                applications_lines.extend(applications.split("\n")[i:])
                break
            if match(positions[index], line):
                applications_lines.append(f"\n(@images:{article_id}_{index}.png)")
                index += 1
                
        main_content_fix = "\n".join(main_content_lines)
        applications_fix = "\n".join(applications_lines)
        with open(f"{output_dir}/MainContent.md", "w", encoding="utf-8") as f:
            f.write(main_content_fix)
        with open(f"{output_dir}/Applications.md", "w", encoding=  "utf-8") as f:
            f.write(applications_fix)
        html_content = self.conver2html(main_content_fix, applications_fix)
        with open(f"{output_dir}/index.html", "w", encoding="utf-8") as f:
            f.write(html_content)
    
    def conver2html(self, main_content, applications):
        full_md = main_content + "\n\n" + applications
        def replace_image_tag(match):
            filename = match.group(1)
            return f"![{filename}]({filename})"
        # 使用正则替换所有 (@images:xxx.png) 格式
        image_pattern = r"\(@images:([^\)]+\.png)\)"
        full_md_with_images = re.sub(image_pattern, replace_image_tag, full_md)
        try:
            html_body = markdown.markdown(
                full_md_with_images,
                extensions=['extra', 'codehilite', 'tables', 'fenced_code']
            )
        except ImportError:
            html_body = "<p>Error: 'markdown' library not installed.</p>"
        # 构造完整 HTML 页面
        html_template = f"""<!DOCTYPE html>
            <html lang="zh">
            <head>
                <meta charset="UTF-8">
                <title>Generated Document</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 10px 0; }}
                    code {{ background-color: #eee; padding: 2px 4px; }}
                    pre {{ background-color: #f4f4f4; padding: 10px; overflow: auto; }}
                </style>
            </head>
            <body>
            {html_body}
            </body>
            </html>"""
        return html_template
    async def produce_propmt(self, prompt: str, output_dir="") -> str:
        """
        阅读文档，从中找出需要插入图片的位置
        Returns:
            position: str - 插入图片的上文内容
            reason: str - 插入图片的原因
            prompt: str - 插入图片的提示词
            negative_prompt: str - 插入图片的负面提示词
        """
        message = [{"role": "user", "content": prompt}]
        try:
            response = await litellm.acompletion(
                model=self.model_gemini_2_5_pro,
                messages=message,
                **self.model_kwargs
            )
            if response.get('choices') and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content']
                content = self.parse_result(content)
                content = json.loads(content)
                return content
            else:
                logger.error("No valid choices found in the response.")
                return ""
        except Exception as e:
            logger.exception(f"Error occurred while calling LLM API: {str(e)}")
            return ""
    
    
    async def produce_image(self, prompt, output_dir="./output", image_name="test.png"):
        """
        调用大模型 API生成图片
        图片自动保存到output_dir目录下
        """
        # 调用litellm库的异步completion接口，传入模型名称和用户消息
        response = await litellm.acompletion(
            model=self.model_gemini_3_pro_image,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
    
        # 遍历API返回的多个图像结果（如果有）
        for i, img in enumerate(response.choices[0].message.images):
            img_url = img["image_url"]["url"]
        
            # 处理base64编码的图像数据（URL可能包含data:image/png;base64,前缀）
            if "," in img_url:
                base64_data = img_url.split(",", 1)[1]
            else:
                print("Invalid image URL format.")
                return None
        
            # 解码并保存图像
            image_data = base64.b64decode(base64_data)
            if i > 0:  
                sub_name = image_name[:-4] + f"_{i}.png" 
            else:
                sub_name = image_name
            file_path = os.path.join(output_dir, sub_name)  
            with open(file_path, "wb") as f:
                f.write(image_data)

            print(f"Image saved to {file_path}")
            
            
    
    
if __name__ == '__main__':
    propmt="生成两张关于气球的图片"
    agent = DrawChemAgent()
    asyncio.run(agent.produce_image(propmt))