<h1 align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist/blob/main/docs/logo_2.png">
    <img src="docs/logo_2.png" width="215" /></a><br>
  <b>AI 科学家：迈向全自动开放式科学发现 🧑‍🔬</b><br>
</h1>

<p align="center">
  📚 <a href="https://arxiv.org/abs/2408.06292">[论文]</a> |
  📝 <a href="https://sakana.ai/ai-scientist/">[博客文章]</a> |
  📂 <a href="https://drive.google.com/drive/folders/1G7A0wTqfXVa-cpexjk0oaXakaSJwffEt">[Google Drive 文件夹]</a>
</p>

人工智能的一个重大挑战是开发能够进行科学研究并发现新知识的智能体。尽管前沿模型已经被用于辅助人类科学家，如头脑风暴或编写代码，但它们仍然需要广泛的人工监督或严重受限于特定任务。

我们很高兴地介绍 **AI 科学家**，这是第一个全面的全自动科学发现系统，使得基础模型如大型语言模型（LLM）能够独立进行研究。

我们进一步提供了论文中所有的运行数据和结果，您可以在[这里](https://drive.google.com/drive/folders/1G7A0wTqfXVa-cpexjk0oaXakaSJwffEt?usp=sharing)访问。我们强烈推荐您阅读一些[Claude 论文](https://drive.google.com/drive/folders/1Mmpz6M1FK4q8e-SewgZcUzdeD0Q2zC39?usp=sharing)（尤其是关于扩散模型的那些），以了解它的优缺点。以下是由 **AI 科学家** 生成的一些示例论文 📝：

1. [DualScale Diffusion: Adaptive Feature Balancing for Low-Dimensional Generative Models](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/adaptive_dual_scale_denoising.pdf)
2. [Multi-scale Grid Noise Adaptation: Enhancing Diffusion Models For Low-dimensional Data](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/grid_based_noise_adaptation.pdf)
3. [GAN-Enhanced Diffusion: Boosting Sample Quality and Diversity](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/gan_diffusion.pdf)
4. [DualDiff: Enhancing Mode Capture in Low-dimensional Diffusion Models via Dual-expert Denoising](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/dual_expert_denoiser.pdf) 
5. [StyleFusion: Adaptive Multi-style Generation in Character-Level Language Models](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/multi_style_adapter.pdf)
6. [Adaptive Learning Rates for Transformers via Q-Learning](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/rl_lr_adaptation.pdf)
8. [Unlocking Grokking: A Comparative Study of Weight Initialization Strategies in Transformer Models](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/weight_initialization_grokking.pdf)
9. [Grokking Accelerated: Layer-wise Learning Rates for Transformer Generalization](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/layerwise_lr_grokking.pdf)
10. [Grokking Through Compression: Unveiling Sudden Generalization via Minimal Description Length](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/mdl_grokking_correlation.pdf)
11. [Accelerating Mathematical Insight: Boosting Grokking Through Strategic Data Augmentation](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/data_augmentation_grokking.pdf)

**注意**：注意！该代码库将执行由 LLM 编写的代码。此类自治涉及各种风险和挑战，包括可能使用危险包、访问网络以及可能生成进程。请根据自己的判断使用。请务必适当容器化并限制网络访问。

<p align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/adaptive_dual_scale_denoising/adaptive_dual_scale_denoising.pdf"><img src="https://github.com/SakanaAI/AI-Scientist/blob/main/docs/anim-ai-scientist.gif" alt="Adaptive Dual Scale Denoising" width="80%" />
</p>

## 目录

1. [需求](#requirements)
2. [运行 AI 科学家论文生成实验](#run-ai-scientist-paper-generation-experiments)
3. [获取 LLM 生成的论文审稿](#getting-an-llm-generated-paper-review)
4. [制作自己的模板](#making-your-own-template)
5. [模板资源](#template-resources)
6. [引用 AI 科学家](#citing-the-ai-scientist)
7. [常见问题](#faq)

## 需求

### 安装

```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# LLM APIs
pip install anthropic aider-chat backoff openai
# Viz
pip install matplotlib pypdf pymupdf4llm
# Install pdflatex
sudo apt-get install texlive-full

# Common Requirements
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

我们为不同的模型使用以下环境变量：

`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`

我们的代码还可以选择使用 Semantic Scholar API Key (`S2_API_KEY`) 来提高吞吐量[如果你有](https://www.semanticscholar.org/product/api)，但原则上它应该在没有 API 密钥的情况下也能正常工作。

请确保为您运行的模型提供相应的密钥，例如：

```
export OPENAI_API_KEY="YOUR KEY HERE"
export S2_API_KEY="YOUR KEY HERE"
```

### 设置 NanoGPT

```bash
# 准备 NanoGPT 数据
python data/enwik8/prepare.py
python data/shakespeare_char/prepare.py
python data/text8/prepare.py
```

#### 创建基线运行（取决于机器）

```
# 设置 NanoGPT 基线运行
# 注意：您必须先运行上面的准备脚本！
cd templates/nanoGPT && python experiment.py --out_dir run_0 && python plot.py
```

#### 创建 NanoGPT_lite 基线运行。我们使用它进行健全性检查
```
# 注意：您必须先运行上面的准备脚本！
cd templates/nanoGPT_lite && python experiment.py --out_dir run_0 && python plot.py
```

### 设置 2D Diffusion

```bash
# 设置 2D Diffusion
git clone https://github.com/gregversteeg/NPEET.git
cd NPEET
pip install .
pip install scikit-learn

# 设置 2D Diffusion 基线运行
cd templates/2d_diffusion && python experiment.py --out_dir run_0 && python plot.py
```

### 设置 Grokking

```bash
# 设置 Grokking 基线运行
cd templates/grokking && python experiment.py --out_dir run_0 && python plot.py
```


## 运行 AI 科学家论文生成实验

**注意：**请确保已完成上述设置步骤。

```bash
conda activate ai_scientist
# 运行论文生成。
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment nanoGPT_lite --num-ideas 2
python launch_scientist.py --model "claude-3-5-sonnet-20240620" --experiment nanoGPT_lite --num-ideas 2
```

## 获取 LLM 生成的论文审稿

```python
import openai
from ai_scientist.perform_review import load_paper, perform_review

client = openai.OpenAI()
model = "gpt-4o-2024-05-13"

# 从 PDF 文件加载论文（原始文本）
paper_txt = load_paper("report.pdf")
# 获取审稿的字典结果
review = perform_review(
    paper_txt,
    model,
    client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,
    temperature=0.1,
)

# 检查审稿结果
review["Overall"]  # 整体评分 1-10
review["Decision"]  # ['接受', '拒绝']
review["Weaknesses"]  # 弱点列表（字符串）
```

要运行批量分析：

```bash
cd review_iclr_bench
python iclr_analysis.py --num_reviews 500  --batch_size 100 --num_fs_examples 1 --num_reflections 5 --temperature 0.1 --num_reviews_ensemble 5
```

## 创建你自己的模板

如果有你希望**AI科学家**探索的研究领域，创建你自己的模板应该非常简单。通常情况下，可以遵循现有模板的结构，其包含以下部分：

- `experiment.py` -- 这是包含主要内容的单个文件。它接受一个`out_dir`参数，该参数指定它应创建文件夹并保存运行过程中生成的相关信息。
- `plot.py` -- 该文件应获取`run`文件夹中的信息并生成图表。代码应清晰易于编辑。
- `prompt.json` -- 在此文件中填写模板的相关信息。
- `seed_ideas.json` -- 在此文件中填写示例创意。你也可以尝试在没有任何示例的情况下生成创意，然后选择一两个最佳创意放在这里。
- `latex/template.tex` -- 我们建议使用我们的latex文件夹，但请确保将预加载的引用替换为你认为更相关的引用。

## 模板资源

我们提供了3个模板，这些模板大量使用了其他代码库中的代码，在此我们表示感谢。（通常，我们会在文件中注明这些信息，但考虑到这可能会影响AI科学家的运行效果，因此在此处表示感谢。）

NanoGPT模板使用了来自[NanoGPT](https://github.com/karpathy/nanoGPT)和[此PR](https://github.com/karpathy/nanoGPT/pull/254)的代码。

2D Diffusion模板使用了来自[tiny-diffusion](https://github.com/tanelp/tiny-diffusion)和[ema-pytorch](https://github.com/lucidrains/ema-pytorch)的代码。

Grokking模板使用了来自[Sea-Snell/grokking](https://github.com/Sea-Snell/grokking)和[danielmamay/grokking](https://github.com/danielmamay/grokking)的代码。

我们感谢这些开源模型和软件包的开发人员，感谢他们的贡献并将他们的工作公开提供。

## 引用 AI 科学家

如果您发现我们的工作有用，请考虑引用我们的工作！

```
@article{lu2024aiscientist,
  title={The {AI} {S}cientist: Towards Fully Automated Open-Ended Scientific Discovery},
  author={Lu, Chris and Lu, Cong and Lange, Robert Tjarko and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2408.06292},
  year={2024}
}
```

## 常见问题 (FAQ)

我们建议您首先阅读我们的论文，了解关于 AI 科学家 (The AI Scientist) 的任何问题。

### 为什么在运行 AI 科学家时缺少文件？
请确保在运行主要实验脚本之前，已完成所有设置和准备步骤。

### 为什么没有生成 PDF 或审稿？
AI 科学家完成一个创意的成功率取决于模板、基础模型以及创意的复杂性。我们建议您参考我们的主要论文。使用 Claude Sonnet 3.5 时观察到的成功率最高。建议使用 GPT-4o 进行审稿，其他模型可能存在积极性偏向或无法符合要求的输出问题。

### 生成每个创意的成本是多少？
通常每篇论文使用 Claude Sonnet 3.5 的成本不到 $15。我们推荐使用 DeepSeek Coder V2 作为更具成本效益的方法。寻找新模型的好地方是 [Aider 排行榜](https://aider.chat/docs/leaderboards/)。

### 如何更改与写作关联的基础会议格式？
更改每个模板中的基础 `template.tex` 文件。

### 如何在不同学科领域运行 AI 科学家？
请参考不同模板的说明。在当前版本中，这仅限于可以用代码表达的创意。然而，解除此限制将是一个令人兴奋的未来工作方向！ :)
