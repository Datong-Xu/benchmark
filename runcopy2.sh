#!/usr/bin/env bash
# ============================================================================
#  Benchmark Evaluation Runner
#  一键运行所有评测。在此文件中统一配置模型信息和评测参数。
# ============================================================================
set -euo pipefail

# ============================================================================
#  [模型配置]  Model Configuration
# ============================================================================
MODEL_PATH="/wekafs/datongxu/flame/exp/hgrn2-340M-10B"
DEVICE="cuda"                  # cuda / cpu

# ============================================================================
#  [评测开关]  Benchmark Toggles  (1 = 运行, 0 = 跳过)
#  后面是单卡MI325X上进行340M参数模型测试花费的时间，总计约3小时
# ============================================================================
EVAL_SPEED=1                   # 前向/反向推理速度测试                        ~1min
EVAL_MMLU=1                    # 英文多学科知识与推理 (57科, 14k题, 选择题)    ~4min
EVAL_HELLASWAG=1               # 英文常识推理与续写合理性 (10k题, 选择题)      ~20min
EVAL_ARC_CHALLENGE=1           # 英文科学推理-困难 (1.2k题, 选择题)           ~2min
EVAL_ARC_EASY=1                # 英文科学推理-简单 (2.4k题, 选择题)           ~3min
EVAL_TRUTHFUL_QA=1             # 英文事实性与抗幻觉 (817题, MC1)              ~2min
EVAL_GSM8K=1                   # 英文小学数学多步推理 (1.3k题, 生成式)         ~90min
EVAL_BBH=1                     # 英文高难推理集合 Big-Bench Hard (生成式)     ~8min
EVAL_HUMANEVAL=1               # 代码生成能力 Python (164题, pass@1)         ~45min
EVAL_CEVAL=1                   # 中文多学科考试 C-Eval (选择题)               ~1min
EVAL_CLUE=1                    # 中文语义相似度 CLUE-AFQMC (句对分类)         ~3min

# ============================================================================
#  [Speed 参数]
# ============================================================================
SPEED_BATCH_SIZE=8
SPEED_SEQ_LEN="512,1024,2048,4096"   # 逗号分隔, 测试多种序列长度
SPEED_WARMUP=5
SPEED_MEASURE=20
SPEED_DTYPE="bfloat16"         # float32 / float16 / bfloat16

# ============================================================================
#  [MMLU 参数]
# ============================================================================
MMLU_FEW_SHOT=5
MMLU_BATCH_SIZE=8

# ============================================================================
#  [HellaSwag 参数]
# ============================================================================
HELLASWAG_BATCH_SIZE=8

# ============================================================================
#  [ARC 参数]
# ============================================================================
ARC_C_FEW_SHOT=25
ARC_E_FEW_SHOT=25

# ============================================================================
#  [GSM8K 参数]
# ============================================================================
GSM8K_FEW_SHOT=8
GSM8K_BATCH_SIZE=4
GSM8K_MAX_NEW_TOKENS=512

# ============================================================================
#  [Big-Bench Hard 参数]
# ============================================================================
BBH_FEW_SHOT=3
BBH_MAX_NEW_TOKENS=64

# ============================================================================
#  [HumanEval 参数]
# ============================================================================
HUMANEVAL_MAX_NEW_TOKENS=512

# ============================================================================
#  [C-Eval 参数]
# ============================================================================
CEVAL_FEW_SHOT=5

# ============================================================================
#  [CLUE-AFQMC 参数]
# ============================================================================
CLUE_FEW_SHOT=8

# ============================================================================
#  以下内容无需修改
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULT_DIR="${MODEL_PATH}/benchmark_results"
mkdir -p "${RESULT_DIR}"

# 统计需要运行的总数
TOTAL=0
[ "${EVAL_SPEED}"         -eq 1 ] && TOTAL=$((TOTAL + 1))
[ "${EVAL_MMLU}"          -eq 1 ] && TOTAL=$((TOTAL + 1))
[ "${EVAL_HELLASWAG}"     -eq 1 ] && TOTAL=$((TOTAL + 1))
[ "${EVAL_ARC_CHALLENGE}" -eq 1 ] && TOTAL=$((TOTAL + 1))
[ "${EVAL_ARC_EASY}"      -eq 1 ] && TOTAL=$((TOTAL + 1))
[ "${EVAL_TRUTHFUL_QA}"   -eq 1 ] && TOTAL=$((TOTAL + 1))
[ "${EVAL_GSM8K}"         -eq 1 ] && TOTAL=$((TOTAL + 1))
[ "${EVAL_BBH}"           -eq 1 ] && TOTAL=$((TOTAL + 1))
[ "${EVAL_HUMANEVAL}"     -eq 1 ] && TOTAL=$((TOTAL + 1))
[ "${EVAL_CEVAL}"         -eq 1 ] && TOTAL=$((TOTAL + 1))
[ "${EVAL_CLUE}"          -eq 1 ] && TOTAL=$((TOTAL + 1))

echo "============================================================"
echo " Benchmark Runner"
echo "============================================================"
echo "  Model:       ${MODEL_PATH}"
echo "  Device:      ${DEVICE}"
echo "  Results dir: ${RESULT_DIR}"
echo "------------------------------------------------------------"
echo "  Speed:         $([ ${EVAL_SPEED}         -eq 1 ] && echo ON || echo OFF)"
echo "  MMLU:          $([ ${EVAL_MMLU}          -eq 1 ] && echo ON || echo OFF)"
echo "  HellaSwag:     $([ ${EVAL_HELLASWAG}     -eq 1 ] && echo ON || echo OFF)"
echo "  ARC-Challenge: $([ ${EVAL_ARC_CHALLENGE} -eq 1 ] && echo ON || echo OFF)"
echo "  ARC-Easy:      $([ ${EVAL_ARC_EASY}      -eq 1 ] && echo ON || echo OFF)"
echo "  TruthfulQA:    $([ ${EVAL_TRUTHFUL_QA}   -eq 1 ] && echo ON || echo OFF)"
echo "  GSM8K:         $([ ${EVAL_GSM8K}         -eq 1 ] && echo ON || echo OFF)"
echo "  BBH:           $([ ${EVAL_BBH}           -eq 1 ] && echo ON || echo OFF)"
echo "  HumanEval:     $([ ${EVAL_HUMANEVAL}     -eq 1 ] && echo ON || echo OFF)"
echo "  C-Eval:        $([ ${EVAL_CEVAL}         -eq 1 ] && echo ON || echo OFF)"
echo "  CLUE:          $([ ${EVAL_CLUE}          -eq 1 ] && echo ON || echo OFF)"
echo "  Total tasks:   ${TOTAL}"
echo "============================================================"
echo ""

PASS=0
FAIL=0
CURR=0
GLOBAL_START=$(date +%s)

# 时间记录数组
declare -a TIME_NAMES=()
declare -a TIME_SECS=()
declare -a TIME_STATUS=()

# 格式化秒数为 HH:MM:SS 或 MM:SS
fmt_time() {
    local sec=$1
    if [ "${sec}" -ge 3600 ]; then
        printf "%dh%02dm%02ds" $((sec/3600)) $((sec%3600/60)) $((sec%60))
    elif [ "${sec}" -ge 60 ]; then
        printf "%dm%02ds" $((sec/60)) $((sec%60))
    else
        printf "%ds" "${sec}"
    fi
}

# ------------------------------------------------------------------
#  Speed
# ------------------------------------------------------------------
if [ "${EVAL_SPEED}" -eq 1 ]; then
    CURR=$((CURR + 1))
    echo ">>> [${CURR}/${TOTAL}] Running Speed benchmark ..."
    _t0=$(date +%s)
    if python "${SCRIPT_DIR}/speed.py" \
        --model_path    "${MODEL_PATH}" \
        --batch_size    "${SPEED_BATCH_SIZE}" \
        --seq_len       "${SPEED_SEQ_LEN}" \
        --warmup_steps  "${SPEED_WARMUP}" \
        --measure_steps "${SPEED_MEASURE}" \
        --dtype         "${SPEED_DTYPE}" \
        --device        "${DEVICE}" \
        --output        "${RESULT_DIR}/speed_results.json"; then
        _s="PASS"; PASS=$((PASS + 1))
    else _s="FAIL"; FAIL=$((FAIL + 1)); fi
    _dt=$(( $(date +%s) - _t0 ))
    echo ">>> Speed ${_s}  ($(fmt_time ${_dt}))"
    TIME_NAMES+=("Speed"); TIME_SECS+=("${_dt}"); TIME_STATUS+=("${_s}")
    echo ""
else echo ">>> Speed skipped (0)"; echo ""; fi

# ------------------------------------------------------------------
#  MMLU
# ------------------------------------------------------------------
if [ "${EVAL_MMLU}" -eq 1 ]; then
    CURR=$((CURR + 1))
    echo ">>> [${CURR}/${TOTAL}] Running MMLU evaluation ..."
    _t0=$(date +%s)
    if python "${SCRIPT_DIR}/MMLU/eval_mmlu.py" \
        --model_path   "${MODEL_PATH}" \
        --data_dir     "${SCRIPT_DIR}/MMLU" \
        --num_few_shot "${MMLU_FEW_SHOT}" \
        --batch_size   "${MMLU_BATCH_SIZE}" \
        --device       "${DEVICE}" \
        --output       "${RESULT_DIR}/mmlu_results.json"; then
        _s="PASS"; PASS=$((PASS + 1))
    else _s="FAIL"; FAIL=$((FAIL + 1)); fi
    _dt=$(( $(date +%s) - _t0 ))
    echo ">>> MMLU ${_s}  ($(fmt_time ${_dt}))"
    TIME_NAMES+=("MMLU"); TIME_SECS+=("${_dt}"); TIME_STATUS+=("${_s}")
    echo ""
else echo ">>> MMLU skipped (0)"; echo ""; fi

# ------------------------------------------------------------------
#  HellaSwag
# ------------------------------------------------------------------
if [ "${EVAL_HELLASWAG}" -eq 1 ]; then
    CURR=$((CURR + 1))
    echo ">>> [${CURR}/${TOTAL}] Running HellaSwag evaluation ..."
    _t0=$(date +%s)
    if python "${SCRIPT_DIR}/hellaswag/eval_hellaswag.py" \
        --model_path  "${MODEL_PATH}" \
        --data_dir    "${SCRIPT_DIR}/hellaswag" \
        --batch_size  "${HELLASWAG_BATCH_SIZE}" \
        --device      "${DEVICE}" \
        --output      "${RESULT_DIR}/hellaswag_results.json"; then
        _s="PASS"; PASS=$((PASS + 1))
    else _s="FAIL"; FAIL=$((FAIL + 1)); fi
    _dt=$(( $(date +%s) - _t0 ))
    echo ">>> HellaSwag ${_s}  ($(fmt_time ${_dt}))"
    TIME_NAMES+=("HellaSwag"); TIME_SECS+=("${_dt}"); TIME_STATUS+=("${_s}")
    echo ""
else echo ">>> HellaSwag skipped (0)"; echo ""; fi

# ------------------------------------------------------------------
#  ARC-Challenge
# ------------------------------------------------------------------
if [ "${EVAL_ARC_CHALLENGE}" -eq 1 ]; then
    CURR=$((CURR + 1))
    echo ">>> [${CURR}/${TOTAL}] Running ARC-Challenge evaluation ..."
    _t0=$(date +%s)
    if python "${SCRIPT_DIR}/ARC-Challenge/eval_arc_challenge.py" \
        --model_path   "${MODEL_PATH}" \
        --data_dir     "${SCRIPT_DIR}/ARC-Challenge" \
        --num_few_shot "${ARC_C_FEW_SHOT}" \
        --device       "${DEVICE}" \
        --output       "${RESULT_DIR}/arc_challenge_results.json"; then
        _s="PASS"; PASS=$((PASS + 1))
    else _s="FAIL"; FAIL=$((FAIL + 1)); fi
    _dt=$(( $(date +%s) - _t0 ))
    echo ">>> ARC-Challenge ${_s}  ($(fmt_time ${_dt}))"
    TIME_NAMES+=("ARC-Challenge"); TIME_SECS+=("${_dt}"); TIME_STATUS+=("${_s}")
    echo ""
else echo ">>> ARC-Challenge skipped (0)"; echo ""; fi

# ------------------------------------------------------------------
#  ARC-Easy
# ------------------------------------------------------------------
if [ "${EVAL_ARC_EASY}" -eq 1 ]; then
    CURR=$((CURR + 1))
    echo ">>> [${CURR}/${TOTAL}] Running ARC-Easy evaluation ..."
    _t0=$(date +%s)
    if python "${SCRIPT_DIR}/ARC-Easy/eval_arc_easy.py" \
        --model_path   "${MODEL_PATH}" \
        --data_dir     "${SCRIPT_DIR}/ARC-Easy" \
        --num_few_shot "${ARC_E_FEW_SHOT}" \
        --device       "${DEVICE}" \
        --output       "${RESULT_DIR}/arc_easy_results.json"; then
        _s="PASS"; PASS=$((PASS + 1))
    else _s="FAIL"; FAIL=$((FAIL + 1)); fi
    _dt=$(( $(date +%s) - _t0 ))
    echo ">>> ARC-Easy ${_s}  ($(fmt_time ${_dt}))"
    TIME_NAMES+=("ARC-Easy"); TIME_SECS+=("${_dt}"); TIME_STATUS+=("${_s}")
    echo ""
else echo ">>> ARC-Easy skipped (0)"; echo ""; fi

# ------------------------------------------------------------------
#  TruthfulQA
# ------------------------------------------------------------------
if [ "${EVAL_TRUTHFUL_QA}" -eq 1 ]; then
    CURR=$((CURR + 1))
    echo ">>> [${CURR}/${TOTAL}] Running TruthfulQA evaluation ..."
    _t0=$(date +%s)
    if python "${SCRIPT_DIR}/truthful_qa/eval_truthful_qa.py" \
        --model_path "${MODEL_PATH}" \
        --data_dir   "${SCRIPT_DIR}/truthful_qa" \
        --device     "${DEVICE}" \
        --output     "${RESULT_DIR}/truthful_qa_results.json"; then
        _s="PASS"; PASS=$((PASS + 1))
    else _s="FAIL"; FAIL=$((FAIL + 1)); fi
    _dt=$(( $(date +%s) - _t0 ))
    echo ">>> TruthfulQA ${_s}  ($(fmt_time ${_dt}))"
    TIME_NAMES+=("TruthfulQA"); TIME_SECS+=("${_dt}"); TIME_STATUS+=("${_s}")
    echo ""
else echo ">>> TruthfulQA skipped (0)"; echo ""; fi

# ------------------------------------------------------------------
#  GSM8K
# ------------------------------------------------------------------
if [ "${EVAL_GSM8K}" -eq 1 ]; then
    CURR=$((CURR + 1))
    echo ">>> [${CURR}/${TOTAL}] Running GSM8K evaluation ..."
    _t0=$(date +%s)
    if python "${SCRIPT_DIR}/gsm8k/eval_gsm8k.py" \
        --model_path     "${MODEL_PATH}" \
        --data_dir       "${SCRIPT_DIR}/gsm8k" \
        --num_few_shot   "${GSM8K_FEW_SHOT}" \
        --batch_size     "${GSM8K_BATCH_SIZE}" \
        --max_new_tokens "${GSM8K_MAX_NEW_TOKENS}" \
        --device         "${DEVICE}" \
        --output         "${RESULT_DIR}/gsm8k_results.json"; then
        _s="PASS"; PASS=$((PASS + 1))
    else _s="FAIL"; FAIL=$((FAIL + 1)); fi
    _dt=$(( $(date +%s) - _t0 ))
    echo ">>> GSM8K ${_s}  ($(fmt_time ${_dt}))"
    TIME_NAMES+=("GSM8K"); TIME_SECS+=("${_dt}"); TIME_STATUS+=("${_s}")
    echo ""
else echo ">>> GSM8K skipped (0)"; echo ""; fi

# ------------------------------------------------------------------
#  Big-Bench Hard
# ------------------------------------------------------------------
if [ "${EVAL_BBH}" -eq 1 ]; then
    CURR=$((CURR + 1))
    echo ">>> [${CURR}/${TOTAL}] Running BBH evaluation ..."
    _t0=$(date +%s)
    if python "${SCRIPT_DIR}/big_bench_hard/eval_bbh.py" \
        --model_path     "${MODEL_PATH}" \
        --data_dir       "${SCRIPT_DIR}/big_bench_hard" \
        --num_few_shot   "${BBH_FEW_SHOT}" \
        --max_new_tokens "${BBH_MAX_NEW_TOKENS}" \
        --device         "${DEVICE}" \
        --output         "${RESULT_DIR}/bbh_results.json"; then
        _s="PASS"; PASS=$((PASS + 1))
    else _s="FAIL"; FAIL=$((FAIL + 1)); fi
    _dt=$(( $(date +%s) - _t0 ))
    echo ">>> BBH ${_s}  ($(fmt_time ${_dt}))"
    TIME_NAMES+=("BBH"); TIME_SECS+=("${_dt}"); TIME_STATUS+=("${_s}")
    echo ""
else echo ">>> BBH skipped (0)"; echo ""; fi

# ------------------------------------------------------------------
#  HumanEval
# ------------------------------------------------------------------
if [ "${EVAL_HUMANEVAL}" -eq 1 ]; then
    CURR=$((CURR + 1))
    echo ">>> [${CURR}/${TOTAL}] Running HumanEval evaluation ..."
    _t0=$(date +%s)
    if python "${SCRIPT_DIR}/humaneval/eval_humaneval.py" \
        --model_path     "${MODEL_PATH}" \
        --data_dir       "${SCRIPT_DIR}/humaneval" \
        --max_new_tokens "${HUMANEVAL_MAX_NEW_TOKENS}" \
        --device         "${DEVICE}" \
        --output         "${RESULT_DIR}/humaneval_results.json"; then
        _s="PASS"; PASS=$((PASS + 1))
    else _s="FAIL"; FAIL=$((FAIL + 1)); fi
    _dt=$(( $(date +%s) - _t0 ))
    echo ">>> HumanEval ${_s}  ($(fmt_time ${_dt}))"
    TIME_NAMES+=("HumanEval"); TIME_SECS+=("${_dt}"); TIME_STATUS+=("${_s}")
    echo ""
else echo ">>> HumanEval skipped (0)"; echo ""; fi

# ------------------------------------------------------------------
#  C-Eval
# ------------------------------------------------------------------
if [ "${EVAL_CEVAL}" -eq 1 ]; then
    CURR=$((CURR + 1))
    echo ">>> [${CURR}/${TOTAL}] Running C-Eval evaluation ..."
    _t0=$(date +%s)
    if python "${SCRIPT_DIR}/ceval-exam/eval_ceval.py" \
        --model_path   "${MODEL_PATH}" \
        --data_dir     "${SCRIPT_DIR}/ceval-exam" \
        --num_few_shot "${CEVAL_FEW_SHOT}" \
        --device       "${DEVICE}" \
        --output       "${RESULT_DIR}/ceval_results.json"; then
        _s="PASS"; PASS=$((PASS + 1))
    else _s="FAIL"; FAIL=$((FAIL + 1)); fi
    _dt=$(( $(date +%s) - _t0 ))
    echo ">>> C-Eval ${_s}  ($(fmt_time ${_dt}))"
    TIME_NAMES+=("C-Eval"); TIME_SECS+=("${_dt}"); TIME_STATUS+=("${_s}")
    echo ""
else echo ">>> C-Eval skipped (0)"; echo ""; fi


# ------------------------------------------------------------------
#  CLUE-AFQMC
# ------------------------------------------------------------------
if [ "${EVAL_CLUE}" -eq 1 ]; then
    CURR=$((CURR + 1))
    echo ">>> [${CURR}/${TOTAL}] Running CLUE-AFQMC evaluation ..."
    _t0=$(date +%s)
    if python "${SCRIPT_DIR}/clue/eval_clue.py" \
        --model_path   "${MODEL_PATH}" \
        --data_dir     "${SCRIPT_DIR}/clue" \
        --num_few_shot "${CLUE_FEW_SHOT}" \
        --device       "${DEVICE}" \
        --output       "${RESULT_DIR}/clue_results.json"; then
        _s="PASS"; PASS=$((PASS + 1))
    else _s="FAIL"; FAIL=$((FAIL + 1)); fi
    _dt=$(( $(date +%s) - _t0 ))
    echo ">>> CLUE ${_s}  ($(fmt_time ${_dt}))"
    TIME_NAMES+=("CLUE"); TIME_SECS+=("${_dt}"); TIME_STATUS+=("${_s}")
    echo ""
else echo ">>> CLUE skipped (0)"; echo ""; fi

# ------------------------------------------------------------------
#  Summary
# ------------------------------------------------------------------
GLOBAL_END=$(date +%s)
GLOBAL_ELAPSED=$(( GLOBAL_END - GLOBAL_START ))

echo "============================================================"
echo " All benchmarks finished.  Passed: ${PASS}  Failed: ${FAIL}"
echo " Total time: $(fmt_time ${GLOBAL_ELAPSED})"
echo " Results directory: ${RESULT_DIR}"
echo "============================================================"

# 时间汇总表
if [ ${#TIME_NAMES[@]} -gt 0 ]; then
    echo ""
    printf "  %-20s %10s  %s\n" "Benchmark" "Time" "Status"
    printf "  %-20s %10s  %s\n" "--------------------" "----------" "------"
    for i in "${!TIME_NAMES[@]}"; do
        printf "  %-20s %10s  %s\n" "${TIME_NAMES[$i]}" "$(fmt_time ${TIME_SECS[$i]})" "${TIME_STATUS[$i]}"
    done
    printf "  %-20s %10s\n" "--------------------" "----------"
    printf "  %-20s %10s\n" "TOTAL" "$(fmt_time ${GLOBAL_ELAPSED})"
    echo ""
fi

if [ -d "${RESULT_DIR}" ]; then
    echo "Result files:"
    ls -lh "${RESULT_DIR}/"
fi

if [ "${FAIL}" -gt 0 ]; then
    exit 1
fi
