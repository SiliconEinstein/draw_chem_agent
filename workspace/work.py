import os,re,json,sys
import litellm
import asyncio
import markdown
from openai import OpenAI
import asyncio
from draw_chem_agent import DrawChemAgent

# prompts = [
#     {
#         "context": "[点缺陷](@entry_id:136257)主要分为三种[基本类](@entry_id:158335)型：**空位 (vacancy)**、**间隙 (interstitial)** 和 **替代 (substitutional)** 缺陷。为了清晰地理解这些概念，我们可以考虑一个理想化的二元[离子晶体](@entry_id:138598)，例如具有[岩盐结构](@entry_id:192480)的 $A^{+}B^{-}$ 型固体（如氯化钠 $\mathrm{NaCl}$），其中阳离子和阴离子各自占据着互穿的面心立方（FCC）亚[晶格](@entry_id:196752) 。-   **空位** 是指[晶格](@entry_id:196752)中一个本应被原子或离子占据的位置出现了空缺。在我们的 $A^{+}B^{-}$ 晶体中，如果一个 $A^{+}$ 离子从其[晶格](@entry_id:196752)位置上移除，就形成了一个阳离子空位。同样，移除一个 $B^{-}$ 离子则形成一个[阴离子空位](@entry_id:161011)。空位的形成在拓扑上移除了一个[晶格](@entry_id:196752)节点，并会降低晶体的宏观密度。\n-   **[间隙缺陷](@entry_id:180338)** 是指一个额外的原子或离子占据了晶体中通常不被占据的位置，即**间隙位置 (interstitial site)**。在[岩盐结构](@entry_id:192480)中，所有八面体空隙都已被离子占据，成为[晶格](@entry_id:196752)的常规位置。因此，可供额外离子占据的最小[间隙位置](@entry_id:149035)是四面体空隙。一个额外的 $A^{+}$ 离子挤入四面体间隙就形成了一个阳离子[间隙缺陷](@entry_id:180338)。由于阴离子通常比阳离子尺寸大得多，[阴离子间隙](@entry_id:156621)缺陷在能量上非常不利，因此更为罕见。\n-   **替代缺陷** 是指一个外来（杂质）原子或离子占据了主体[晶格](@entry_id:196752)中的一个常规位置。例如，在一个 $A^{+}B^{-}$ 晶体中，一个二价阳离子 $M^{2+}$ 可能会替代一个 $A^{+}$ 离子。这类缺陷通常是通过在晶体生长或后续处理过程中有意或无意地引入杂质（**掺杂**）而产生的。",
#         # "position": "点缺陷主要分为三种基本类型：**空位 (vacancy)**、**间隙 (interstitial)** 和 **替代 (substitutional)** 缺陷。",
#         "reason": "该位置首次引入点缺陷的三种基本类型，配图能够直观地展示这三种缺陷在晶格中的形态，帮助读者建立核心概念的视觉印象。图片将并列展示一个完美晶格和三种含有不同缺陷的晶格，形成清晰的对比。",
#         # "prompt": "Create a clean, academic-style schematic diagram in four panels. Use a simple 2D square lattice of blue spheres to represent atoms. Panel 1 (top left) is labeled 'Perfect Crystal' and shows a perfect 4x4 grid of blue spheres. Panel 2 (top right) is labeled 'Vacancy' and shows the same grid with one blue sphere missing from the center. Panel 3 (bottom left) is labeled 'Interstitial Defect' and shows the grid with an extra, smaller red sphere squeezed between the central blue spheres. Panel 4 (bottom right) is labeled 'Substitutional Defect' and shows the grid with the central blue sphere replaced by a larger green sphere. The style should be minimalist, with clean lines, a light gray background, and clear, sans-serif labels. Use a scientific illustration aesthetic.",
#         # "negative_prompt": "photorealistic, 3D rendering, complex shadows, dark background, handwritten text, cartoonish, blurry, cluttered, organic shapes."
#     },
#     {
#         "context": "让我们用 [Kröger-Vink 记法](@entry_id:149058)来描述 $A^{+}B^{-}$ 晶体中的缺陷：\n-   **完美[晶格](@entry_id:196752)**：一个 $A^{+}$ 离子占据一个阳离子位点，其有效电荷为 $(+1) - (+1) = 0$，记为 $A_A^\times$。同样，一个 $B^{-}$ 离子占据一个阴离子位点，记为 $B_B^\times$。\n-   **空位**：移除一个 $A^{+}$ 离子形成一个阳离子空位 $V_A$。该位置现在是空的（[电荷](@entry_id:275494)为0），因此其[有效电荷](@entry_id:748807)为 $0 - (+1) = -1$。记为 $V_A^\prime$。类似地，移除一个 $B^{-}$ 离子形成的[阴离子空位](@entry_id:161011) $V_B$ 的[有效电荷](@entry_id:748807)为 $0 - (-1) = +1$，记为 $V_B^\bullet$。\n-   **[间隙缺陷](@entry_id:180338)**：一个间隙位置在完美[晶格](@entry_id:196752)中是空的，因此其参考[电荷](@entry_id:275494)为 0。当一个 $A^{+}$ 离子占据该位置时，其[有效电荷](@entry_id:748807)为 $(+1) - 0 = +1$，记为 $A_i^\bullet$。\n-   **替代缺陷**：当一个二价杂质离子 $M^{2+}$ 替代一个 $A^{+}$ 离子时，该缺陷记为 $M_A$。其有效电荷为 $(+2) - (+1) = +1$，记为 $M_A^\bullet$。\n这个强大的记法构成了**[缺陷化学](@entry_id:158602) (defect chemistry)** 的基础，它允许我们将缺陷的形成和相互作用写成类似于[化学反应](@entry_id:146973)的[平衡方程](@entry_id:172166)，并要求这些方程在质量、[晶格](@entry_id:196752)位置和有效电荷三方面都保持平衡。",
#         # "position": "这个强大的记法构成了**[缺陷化学](@entry_id:158602) (defect chemistry)** 的基础，它允许我们将缺陷的形成和相互作用写成类似于[化学反应](@entry_id:146973)的[平衡方程](@entry_id:172166)，并要求这些方程在质量、[晶格](@entry_id:196752)位置和有效电荷三方面都保持平衡。",
#         "reason": "Kröger-Vink记法是描述缺陷的标准化但抽象的语言。一张图表通过几个关键实例（如阳离子空位、阴离子空位、间隙离子）来图解其表示法 M_S^C 的构成，特别是有效电荷的计算，能够极大地降低理解门槛，使读者能快速掌握这一工具。",
#         # "prompt": "Design an infographic-style table explaining Kröger-Vink Notation for an ionic crystal A+B-. The table should have four rows for four examples. Columns should be: 'Defect Type', 'Schematic', 'Notation (M_S^C)', and 'Effective Charge Calculation'. Example 1: 'Cation Vacancy', a 2D lattice with a missing A+ ion, Notation 'V_A′', Calculation '(0) - (+1) = -1'. Example 2: 'Anion Vacancy', a lattice with a missing B- ion, Notation 'V_B•', Calculation '(0) - (-1) = +1'. Example 3: 'Cation Interstitial', an A+ ion in an interstitial site, Notation 'A_i•', Calculation '(+1) - (0) = +1'. Example 4: 'Substitutional Defect', an M2+ ion on an A+ site, Notation 'M_A•', Calculation '(+2) - (+1) = +1'. Use simple icons for ions (e.g., blue circle for A+, orange for B-, green for M2+). The style must be clean, modern, and educational, with a color-coded scheme and clear typography against a white background. Academic, minimalist aesthetic.",
#         # "negative_prompt": "cluttered, hand-drawn, 3D, photorealistic, dark, confusing layout, excessive text, blurry."
#     },
#     {
#         "context": "### 内禀缺陷平衡：肖特基与弗伦克尔无序\n在任何高于绝对[零度](@entry_id:156285)的温度下，由于熵的贡献，晶体中总会自发地形成一定浓度的[点缺陷](@entry_id:136257)。这些在纯净晶体中由[热力学](@entry_id:141121)驱动产生的缺陷称为**内禀缺陷 (intrinsic defects)**。在离子晶体中，为了维持整体的[电中性](@entry_id:157680)，这些缺陷通常成对出现。两种最主要的内禀缺陷类型是肖特基无序和弗伦克尔无序 。#### 肖特基无序 (Schottky Disorder)\n**[肖特基缺陷](@entry_id:138313) (Schottky defect)** 由一对化学计量比的阳离子空位和[阴离子空位](@entry_id:161011)组成。例如，在 $\mathrm{NaCl}$ ($A^{+}B^{-}$ 型) 晶体中，一个[肖特基缺陷](@entry_id:138313)对包含一个钠离子空位 ($V_{\mathrm{Na}}^\prime$) 和一个[氯离子](@entry_id:263601)空位 ($V_{\mathrm{Cl}}^\bullet$)。这种缺陷的形成可以看作是将晶体内部的一个中性[化学式](@entry_id:136318)单元转移到晶体表面，从而在体相中留下空位对.\n使用 [Kröger-Vink 记法](@entry_id:149058)，[肖特基缺陷](@entry_id:138313)的形成反应可以写成两种形式。一种是简化的体相反应式，其中 $\varnothing$ 代表完美的[晶格](@entry_id:196752)：\n$$ \varnothing \rightleftharpoons V_A^\prime + V_B^\bullet $$\n这个表达式强调了在体相中生成了等量的、带相反有效电荷的空位，从而保持了[电中性](@entry_id:157680)。\n另一种更完整的表达式明确地体现了与[晶体表面](@entry_id:195760)的物质交换 ：\n$$ \mathrm{Na_{Na}^{\times}} + \mathrm{Cl_{Cl}^{\times}} \rightleftharpoons V_{\mathrm{Na}}^{\prime} + V_{\mathrm{Cl}}^{\bullet} + \mathrm{NaCl}_{(\text{surf})} $$\n这里，$\mathrm{Na_{Na}^{\times}}$ 和 $\mathrm{Cl_{Cl}^{\times}}$ 代表完美的[晶格](@entry_id:196752)位点，而 $\mathrm{NaCl}_{(\text{surf})}$ 表示在晶体表面形成的一个新的、[电中性](@entry_id:157680)的 $\mathrm{NaCl}$ 单元。这个反应清晰地表明，[肖特基缺陷](@entry_id:138313)的形成保持了晶体的化学计量比，但由于从体相中移除了原子，它会导致晶体密度的降低。\n#### 弗伦克尔无序 (Frenkel Disorder)\n**[弗伦克尔缺陷](@entry_id:151563) (Frenkel defect)** 由一个空位和一个同种离子的[间隙缺陷](@entry_id:180338)组成。它通常发生在阳离子身上，因为阳离子尺寸较小，更容易挤入间隙位置。当一个阳离子离开其正常的[晶格](@entry_id:196752)位置，并迁移到附近的一个[间隙位置](@entry_id:149035)时，就形成了一个**阳离子[弗伦克尔对](@entry_id:749586) (cation Frenkel pair)**。\n其形成反应可以写作：\n$$ A_A^\times \rightleftharpoons V_A^\prime + A_i^\bullet $$\n在这个过程中，一个在 $A$ 位点上的中性 $A$ 离子 ($A_A^\times$) 转变为一个带负[有效电荷](@entry_id:748807)的空位 ($V_A^\prime$) 和一个带正有效电荷的间隙离子 ($A_i^\bullet$)。总的[有效电荷](@entry_id:748807)变化为零，保持了电中性。由于没有原子离开晶体，[弗伦克尔缺陷](@entry_id:151563)的形成基本不改变晶体的宏观密度。",
#         # "position": "在离子晶体中，为了维持整体的[电中性](@entry_id:157680)，这些缺陷通常成对出现。两种最主要的内禀缺陷类型是肖特基无序和弗伦克尔无序 。",
#         "reason": "此处对比了两种最核心的内禀缺陷：肖特基缺陷和弗伦克尔缺陷。一张并列对比的示意图能清晰地揭示二者在形成机制（原子去向）、对晶格的影响（是否产生间隙）以及对密度的影响上的根本区别，是理解本章内容的关键。",
#         # "prompt": "Create a two-panel, side-by-side comparison diagram with a minimalist, scientific style. Left panel labeled 'Schottky Defect'. It shows a 2D ionic crystal lattice (alternating small blue circles for cations and large red circles for anions). An arrow points from a cation inside the lattice to the crystal surface, and another arrow points from an anion inside to the surface. The lattice shows a resulting cation vacancy and an anion vacancy. Right panel labeled 'Frenkel Defect'. It shows a similar lattice, but an arrow points from a cation's lattice site to a nearby interstitial position. The lattice shows a resulting cation vacancy and an interstitial cation. Both panels should have clean lines, a neutral background, and simple, clear labels. The overall aesthetic is that of a modern textbook illustration.",
#         # "negative_prompt": "photorealistic, complex textures, shadows, 3D effects, dark colors, cartoon style, messy annotations."
#     }
# ]

# async def main():
#     agent = DrawChemAgent()
#     index = 0
#     for prompt in prompts:
#         prompt = json.dumps(prompt)
#         await agent.produce_image(prompt, output_dir="./output", image_name=f"{123}_{index}.png")
#         index += 1
    
# if __name__ == '__main__':
#     asyncio.run(main())

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
        full_text = main_content + "\n" + applications
        full_text_lines = full_text.split("\n")
        full_text_lines = [clean_text(line) for line in full_text_lines if len(line.strip()) > 0]
        # 为每个句子编号
        for i, line in enumerate(full_text_lines):
            full_text_lines[i] = f"{[i + 1]}: {line}"
        full_text = "\n".join(full_text_lines)
        prompt = agent.get_prompt("./prompt", "get_insert_position_v2", {"【ARTICLE】": full_text})
        draw_image_prompts = await agent.produce_propmt(prompt)
        with open(f"{output_path}/draw_image_prompts.txt", "w", encoding="utf-8") as f:
            json.dump(draw_image_prompts, f, indent=4, ensure_ascii=False)
        positions = [img["position"] for img in draw_image_prompts]
        # 在原文中插入图片标记符，并且保存到指定路径
        agent.insert_marker(article_id, positions, main_content, applications, output_path)
        for index, draw_image_prompt in enumerate(draw_image_prompts):
            context_ids = draw_image_prompt["context"]
            context = "\n".join([full_text_lines[id - 1] for id in context_ids])
            await agent.produce_image(context, output_dir=output_path, image_name=f"{article_id}_{index}.png")
        
        
if __name__ == '__main__':
    asyncio.run(main())