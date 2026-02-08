from datasets import load_dataset
from pathlib import Path

cache_dir = Path(__file__).resolve().parent

# ds = load_dataset("cais/mmlu", "all", cache_dir=str(cache_dir))
# ds = load_dataset("openai/gsm8k", "main", cache_dir=str(cache_dir))
# ds = load_dataset("Rowan/hellaswag", cache_dir=str(cache_dir))
# ds = load_dataset("domenicrosati/TruthfulQA", cache_dir=str(cache_dir))
# ds = load_dataset("Joschka/big_bench_hard", "boolean_expressions", cache_dir=str(cache_dir))
# ds = load_dataset("allenai/ai2_arc", "ARC-Easy", cache_dir=str(cache_dir))
# ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir=str(cache_dir))
# ds = load_dataset("openai/openai_humaneval", cache_dir=str(cache_dir))
# ds = load_dataset("ceval/ceval-exam", "accountant", cache_dir=str(cache_dir))
ds = load_dataset("clue/clue", "afqmc", cache_dir=str(cache_dir))

#  MMLU：多学科知识与推理综合
#  GSM8K：小学数学与多步推理
#  HellaSwag：常识与续写合理性
#  TruthfulQA：事实性与抗幻觉
#  BBH（Big‑Bench Hard）：高难推理集合
#  ARC-Easy：科学推理（简单子集）
#  ARC-Challenge：科学推理（困难子集）
#  HumanEval：代码生成能力（若有代码需求）
#  C‑Eval：中文多学科考试
#  CLUE‑AFQMC 或 ChnSentiCorp：中文句子理解/情感分类（补充理解类能力）