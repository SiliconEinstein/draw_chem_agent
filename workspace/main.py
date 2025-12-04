import os,sys,json,re
import asyncio
from draw_chem_agent import DrawChemAgent

def clean_text(text):
    pattern = r"\(@[^:]+:[^)]*\)"
    return re.sub(pattern, "", text)

async def main():
    article_ids = [826804]
    # article_ids = [826750, 826738, 826808, 826793]
    agent = DrawChemAgent()
    for article_id in article_ids:
        output_path = f"./output"
        os.makedirs(output_path, exist_ok=True)
        main_content, applications = agent.get_article(article_id)
        # 为句子编号
        full_text = main_content + "\n" + applications
        full_text_lines = full_text.split("\n")
        full_text_lines = [clean_text(line) for line in full_text_lines if len(line.strip()) > 0]
        for i, line in enumerate(full_text_lines):
            full_text_lines[i] = f"{[i + 1]}: {line}"
        full_text = "\n".join(full_text_lines)
        
        # 获取图片插入位置以及对应的上下文原文的序号
        prompt = agent.get_prompt("./prompt", "get_insert_position", {"【ARTICLE】": full_text})
        draw_image_prompts = await agent.produce_propmt(prompt)
        with open(f"{output_path}/draw_image_prompts.txt", "w", encoding="utf-8") as f:
            json.dump(draw_image_prompts, f, indent=4, ensure_ascii=False)
        
        # 在原文中插入图片标记符，并且保存到指定路径
        positions = [img["position"] for img in draw_image_prompts]
        agent.insert_marker(article_id, positions, main_content, applications, output_path)
        
        # 绘图
        for index, draw_image_prompt in enumerate(draw_image_prompts):
            context_ids = draw_image_prompt["context"]
            context = "\n".join([full_text_lines[id - 1] for id in context_ids])
            draw_image_prompts[index]["context"] = context
            prompt = agent.get_prompt("./prompt", "draw_by_text", {"【CONTEXT】": context})
            await agent.produce_image(prompt, output_dir=output_path, image_name=f"{article_id}_{index}.png")
            
        # 保存上下文原文
        with open(f"{output_path}/draw_image_prompts.txt", "w", encoding="utf-8") as f:
            json.dump(draw_image_prompts, f, indent=4, ensure_ascii=False)
        
        
if __name__ == '__main__':
    asyncio.run(main())
