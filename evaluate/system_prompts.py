"""System prompts for each benchmark task."""

SYSTEM_PROMPTS = {
    "gsm8k": "You are given a math problem. Show your reasoning and provide the final answer as strictly a number in \\boxed{} format.",
    "math500": "You are given a math problem. Show your reasoning and provide the final answer in latex format enclosed in \\boxed{}.",
    "aime": "You are given a math problem. Solve it and provide the final answer strictly as an integer between 0 and 999, enclose it in \\boxed{} format.",
    "mmlu": "You are given a multiple choice question. Answer the question by selecting the correct option (A, B, C, or D). Show your reasoning and return your final answer in \\boxed{} format with only the letter corresponding to the correct answer.",
    "gpqa": "You are given a multiple choice question. Answer the question by selecting the correct option (A, B, C, D, or E). Show your reasoning and return your final answer in \\boxed{} format with only the letter corresponding to the correct answer.",
    "livecodebench": "You are an expert programmer. Write clean, efficient Python code to solve the following programming problem.",
}
