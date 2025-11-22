"""System prompts for each benchmark task."""

SYSTEM_PROMPTS = {
    "gsm8k": "Please reason step by step, and put your final answer within \\boxed{}",
    "math500": "Please reason step by step, and put your final answer within \\boxed{}",
    "aime": "Please reason step by step, and put your final answer within \\boxed{}",
    "mmlu": "Please reason step by step, and select the answer from the given choices A, B, C, or D. Respond only with the letter of the correct answer, from A to D, not with the answer itself. Put the index of the correct answer within \\boxed{}.",
    "gpqa": "Please reason step by step, and select the answer from the given choices A, B, C, or D. Respond only with the letter of the correct answer, from A to D, not with the answer itself. Put the index of the correct answer within \\boxed{}.",
}
