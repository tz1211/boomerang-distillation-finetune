"""System prompts for each benchmark task."""

SYSTEM_PROMPTS = {
    "gsm8k": "Please reason step by step, and put your final answer within \\boxed{}",
    "math500": "Please reason step by step, and put your final answer within \\boxed{}",
    "aime": "Please reason step by step, and put your final answer within \\boxed{}",
    "mmlu": "Please reason step by step, and select the answer from the given choices A, B, C, or D. Respond only with the letter of the correct answer, from A to D, not with the answer itself. Put the index of the correct answer within \\boxed{}.",
    "gpqa": "You are given a multiple choice question. Answer the question by selecting the correct option (A, B, C, D, or E). Show your reasoning and return your final answer in \\boxed{} format with only the letter corresponding to the correct answer.",
    "livecodebench": "You are an expert programmer. Write clean, efficient Python code to solve the following programming problem.",
}
