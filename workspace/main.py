import os,sys,json,re
import asyncio
from draw_chem_agent import DrawChemAgent

def clean_text(text):
    pattern = r"\(@[^:]+:[^)]*\)"
    return re.sub(pattern, "", text)

def eval(agent, article_ids, output_root_path):
    max_tries = 3
    for article_id in article_ids:
        output_path = f"{output_root_path}/{article_id}"
    # root_path = f"./output/{article_id}"
        with open(output_path + "/draw_image_prompts.txt", "r", encoding="utf-8") as f:
            draw_image_prompts = json.load(f)
        eval_results = []
        total_score = 0
        quality_num = 0
        for index, draw_image_prompt in enumerate(draw_image_prompts):
            image_name = draw_image_prompt["image_name"]
            image_path = f"{output_path}/{image_name}"
            if not os.path.exists(image_path):
                continue
            
            prompt = agent.get_prompt("./prompt", "eval_image_v2", {"【CONTEXT】": draw_image_prompt["context"]})
            
            for attempt in range(max_tries):
                try:
                    eval_res = agent.eval_image(image_path, prompt)
                    if eval_res.get("reason") == "" or eval_res.get("describe") == "":
                        print(f"第 {attempt + 1} 次尝试无效，重试中... 图片: {image_path}")
                        continue
                    else:
                        break
                except Exception as e:
                    print(f"An error occurred during evaluation: {e}")
                    continue

            eval_results.append({
                "image_path": image_path,
                "describe": eval_res["describe"],
                "reason": eval_res["reason"],
                "score": eval_res["score"]
            })
            total_score += int(eval_res["score"])
            quality_num += 1 if int(eval_res["score"]) >= 2 else 0

        avg_score = total_score / len(eval_results)
        with open(f"{output_path}/eval_results_v2.md", "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=4, ensure_ascii=False)
        print({
            "quality_num": quality_num,
            "quality_rate": round((quality_num / len(eval_results)) * 100),
            "avg_score": round(avg_score, 2),
        })

async def main(agent, article_ids, output_root_path):
    # article_ids = [826738]
    for article_id in article_ids:
        output_path = f"{output_root_path}/{article_id}"
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
            prompt = agent.get_prompt("./prompt", "draw_by_text", {"【CONTEXT】": context, "【REASON】": draw_image_prompt["reason"]})
            await agent.produce_image(prompt, output_dir=output_path, image_name=f"{article_id}_{index}.png")
            draw_image_prompts[index]["image_name"] = f"{article_id}_{index}.png"
            
        # 保存上下文原文
        with open(f"{output_path}/draw_image_prompts.txt", "w", encoding="utf-8") as f:
            json.dump(draw_image_prompts, f, indent=4, ensure_ascii=False)
        
    # # 自动评分
    # eval_res = eval(agent, article_ids, output_root_path)
    # print(eval_res)
        
if __name__ == '__main__':
    article_ids = [826750, 826738, 826808, 826793]
    # article_ids = [826738]
    output_root_path = "./output/has_reason"
    agent = DrawChemAgent()
    # asyncio.run(main(agent, article_ids, output_root_path))
    eval(agent, article_ids, output_root_path)
