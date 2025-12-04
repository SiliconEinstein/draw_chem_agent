import os,sys,json
import asyncio
from draw_chem_agent import DrawChemAgent

async def main():
    # article_ids = [826804]
    article_ids = [826750, 826738, 826808, 826793]
    agent = DrawChemAgent()
    for article_id in article_ids:
        output_path = f"./output/{article_id}"
        os.makedirs(output_path, exist_ok=True)
        main_content, applications = agent.get_article(article_id)
        full_text = main_content + "\n" + applications
        prompt = agent.get_prompt("./prompt", "get_insert_position", {"【ARTICLE】": full_text})
        draw_image_prompts = await agent.produce_propmt(prompt)
        with open(f"{output_path}/draw_image_prompts.txt", "w", encoding="utf-8") as f:
            json.dump(draw_image_prompts, f, indent=4, ensure_ascii=False)
        positions = [img["position"] for img in draw_image_prompts]
        # 在原文中插入图片标记符，并且保存到指定路径
        agent.insert_marker(article_id, positions, main_content, applications, output_path)
        for index, draw_image_prompt in enumerate(draw_image_prompts):
            draw_image_prompt = json.dumps(draw_image_prompt)
            await agent.produce_image(draw_image_prompt, output_dir=output_path, image_name=f"{article_id}_{index}.png")
        
        
if __name__ == '__main__':
    asyncio.run(main())
