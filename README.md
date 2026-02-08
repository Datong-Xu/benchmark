# LLM Benchmark Suite

一套自包含的语言模型评测工具，覆盖 **11 项基准测试**，涵盖知识推理、数学、代码生成、常识理解、中文能力和推理速度。所有数据集已预下载为 Arrow 格式，无需联网即可运行，一键启动全部评测。

## 目录结构

```
benchmark/
├── run.sh                  # 一键运行所有评测的主脚本
├── speed.py                # 前向/反向推理速度测试
├── download.py             # 数据集下载工具 (使用 HuggingFace datasets)
│
├── MMLU/                   # 英文多学科知识与推理 (57科, 14k题)
│   ├── eval_mmlu.py
│   └── *.arrow
│
├── hellaswag/              # 英文常识推理与续写合理性 (10k题)
│   ├── eval_hellaswag.py
│   └── *.arrow
│
├── ARC-Challenge/          # 英文科学推理 - 困难 (1.2k题)
│   ├── eval_arc_challenge.py
│   └── *.arrow
│
├── ARC-Easy/               # 英文科学推理 - 简单 (2.4k题)
│   ├── eval_arc_easy.py
│   └── *.arrow
│
├── truthful_qa/            # 英文事实性与抗幻觉 (817题)
│   ├── eval_truthful_qa.py
│   └── *.arrow
│
├── gsm8k/                  # 英文小学数学多步推理 (1.3k题)
│   ├── eval_gsm8k.py
│   └── *.arrow
│
├── big_bench_hard/         # 英文高难推理集合 BBH
│   ├── eval_bbh.py
│   └── *.arrow
│
├── humaneval/              # Python 代码生成 (164题, pass@1)
│   ├── eval_humaneval.py
│   └── *.arrow
│
├── ceval-exam/             # 中文多学科考试 C-Eval
│   ├── eval_ceval.py
│   └── *.arrow
│
└── clue/                   # 中文语义相似度 CLUE-AFQMC (4.3k题)
    ├── eval_clue.py
    └── *.arrow
```

## 支持的基准测试

| # | 基准测试 | 语言 | 类型 | 题量 | 评测方式 | 参考时间* |
|---|---------|------|------|------|---------|----------|
| 1 | **Speed** | - | 速度测试 | - | 前向/反向/前向+反向吞吐量 | ~1 min |
| 2 | **MMLU** | EN | 多学科知识 | 14,042 | 5-shot, log-likelihood 选择 | ~4 min |
| 3 | **HellaSwag** | EN | 常识推理 | 10,042 | 0-shot, log-likelihood 续写选择 | ~20 min |
| 4 | **ARC-Challenge** | EN | 科学推理(难) | 1,172 | 25-shot, log-likelihood 选择 | ~2 min |
| 5 | **ARC-Easy** | EN | 科学推理(易) | 2,376 | 25-shot, log-likelihood 选择 | ~3 min |
| 6 | **TruthfulQA** | EN | 事实性/抗幻觉 | 817 | MC1, log-likelihood 选择 | ~2 min |
| 7 | **GSM8K** | EN | 数学推理 | 1,319 | 8-shot, 生成式 (greedy) | ~90 min |
| 8 | **BBH** | EN | 高难推理 | varies | 3-shot, 生成式 (exact match) | ~8 min |
| 9 | **HumanEval** | EN | 代码生成 | 164 | 0-shot, pass@1 (沙盒执行) | ~45 min |
| 10 | **C-Eval** | ZH | 中文多学科 | varies | 5-shot, logit 选择 | ~1 min |
| 11 | **CLUE-AFQMC** | ZH | 语义相似度 | 4,316 | 8-shot, log-likelihood 选择 | ~3 min |

> \* 参考时间基于单卡 MI325X 上 340M 参数模型，总计约 3 小时。

## 快速开始

### 一键运行所有评测

1. 编辑 `run.sh` 顶部的配置：

```bash
MODEL_PATH="/path/to/your/model"   # HuggingFace 格式的模型路径，此文件夹下必须有.safetensor等模型文件
```

2. 通过评测开关选择要运行的测试（`1` = 运行，`0` = 跳过）：

```bash
EVAL_SPEED=1
EVAL_MMLU=1
EVAL_HELLASWAG=1
EVAL_ARC_CHALLENGE=1
EVAL_ARC_EASY=1
EVAL_TRUTHFUL_QA=1
EVAL_GSM8K=1
EVAL_BBH=1
EVAL_HUMANEVAL=1
EVAL_CEVAL=1
EVAL_CLUE=1
```

3. 运行：

```bash
bash run.sh
```

评测结果将保存到 `{MODEL_PATH}/benchmark_results/` 目录下。

### 单独运行某项评测

每个评测脚本也可以独立运行，例如：

```bash
# MMLU
python MMLU/eval_mmlu.py \
    --model_path /path/to/model \
    --data_dir MMLU \
    --num_few_shot 5 \
    --batch_size 8 \
    --output results/mmlu.json

# Speed
python speed.py \
    --model_path /path/to/model \
    --batch_size 8 \
    --seq_len 512,1024,2048,4096 \
    --dtype bfloat16

# GSM8K
python gsm8k/eval_gsm8k.py \
    --model_path /path/to/model \
    --data_dir gsm8k \
    --num_few_shot 8 \
    --max_new_tokens 512

# HumanEval
python humaneval/eval_humaneval.py \
    --model_path /path/to/model \
    --data_dir humaneval \
    --max_new_tokens 512
```

## 评测方法说明

### 选择题类评测 (MMLU, HellaSwag, ARC, TruthfulQA, C-Eval, CLUE)

使用 **log-likelihood** 方法：对每个候选答案，计算其在上下文条件下的条件概率（对数似然），选择概率最高的选项作为模型的预测。这种方法不依赖于模型的生成能力，适合评测基座模型。

### 生成式评测 (GSM8K, BBH)

使用 **greedy decoding** 生成回答，然后从回答中提取答案与标准答案比较。GSM8K 使用数字精确匹配，BBH 使用归一化后的字符串精确匹配。

BBH 答案提取采用多级策略，以兼容不同能力水平的模型：
1. 若生成的第一行本身就是标准短答案（如 `True`/`False`/`Yes`/`No`），直接采用
2. 匹配 `the answer is ...` / `answer: ...` 等常见模式
3. 若第一个词是已知答案词（如 `True`/`False`），仅取第一个词（适配小模型续写行为）
4. 兜底返回完整第一行

可通过环境变量 `BBH_DEBUG=N` 打印前 N 个样本的调试信息（预测值、期望值、原始生成文本），用于诊断答案提取问题：

```bash
BBH_DEBUG=10 bash run.sh
```

### 代码评测 (HumanEval)

模型根据函数签名和 docstring 生成代码补全，在隔离的子进程沙盒中执行测试用例，报告 **pass@1** 指标。

### 速度测试 (Speed)

分三种模式测试：
- **Forward**：纯前向推理（`torch.no_grad`，eval 模式）
- **Backward**：仅反向传播（前向不计时）
- **Forward+Backward**：完整训练步

报告平均耗时（ms/step）、吞吐量（tokens/sec）和 GPU 显存峰值。

## 输出格式

每项评测输出一个 JSON 文件，包含模型路径、准确率、运行时间等信息。`run.sh` 运行结束后会打印汇总表格：

```
  Benchmark              Time  Status
  -------------------- ----------  ------
  Speed                       42s  PASS
  MMLU                      3m28s  PASS
  HellaSwag                19m12s  PASS
  ARC-Challenge             1m45s  PASS
  ...
  --------------------  ----------
  TOTAL                  2h58m30s
```

## 注意事项

由于 GitHub 单文件大小限制（100MB），`MMLU/mmlu-auxiliary_train.arrow`（154MB）**未包含在本仓库中**。该文件是 MMLU 的辅助训练集，**不影响评测运行**（评测仅使用 dev 和 test 集）。

如需获取该文件，请运行以下命令手动下载：

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('cais/mmlu', 'all', cache_dir='MMLU')
"
```

或直接从 HuggingFace Hub 下载：

```bash
# 使用 huggingface-cli
huggingface-cli download cais/mmlu --repo-type dataset --local-dir MMLU
```

## 数据集来源

所有数据集通过 `download.py` 从 HuggingFace Hub 下载，并以 Arrow 格式本地存储：

| 数据集 | HuggingFace 来源 |
|--------|-----------------|
| MMLU | `cais/mmlu` |
| HellaSwag | `Rowan/hellaswag` |
| ARC | `allenai/ai2_arc` |
| TruthfulQA | `domenicrosati/TruthfulQA` |
| GSM8K | `openai/gsm8k` |
| BBH | `Joschka/big_bench_hard` |
| HumanEval | `openai/openai_humaneval` |
| C-Eval | `ceval/ceval-exam` |
| CLUE-AFQMC | `clue/clue` |

## 模型兼容性

支持所有 HuggingFace `AutoModelForCausalLM` 可加载的模型。对于 [fla](https://github.com/sustcsonglin/flash-linear-attention) 系列模型（如 GatedDeltaNet、GLA、HGRN2、DeltaNet 等），脚本内置了自动注册逻辑，无需额外配置。

## License

本仓库中的评测脚本为自研代码。数据集版权归各原始来源所有，请遵循对应的许可协议。
