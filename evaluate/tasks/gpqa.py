"""GPQA-Diamond evaluation - Graduate-Level Google-Proof Q&A Dataset (Diamond subset)."""

import random
from typing import Dict, Optional

from vllm import LLM
from datasets import load_dataset
from transformers import AutoTokenizer

from evaluate.utils.utils import batch_generate, grade_answer, multiple_choice_num_to_letter, extract_last_boxed


def evaluate_gpqa(
    model: LLM,
    tokenizer: AutoTokenizer,
    apply_chat_template: bool = True,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    limit: Optional[int] = None,
    system_prompt: Optional[str] = None,
    log_n_results: int = 100,
) -> Dict:
    """
    Evaluate model on GPQA-Diamond dataset (multiple choice questions).
    
    Args:
        model: The vLLM LLM model
        tokenizer: The tokenizer
        apply_chat_template: Whether to apply chat template
        max_new_tokens: Maximum tokens to generate (should be small for MC)
        temperature: Sampling temperature
        limit: Limit number of examples
        system_prompt: Optional system prompt to prepend to each question
        log_n_results: Number of results to log
    Returns:
        Dictionary with accuracy and detailed results
    """
    print("Loading GPQA-Diamond dataset...")
    
    # Load GPQA-Diamond dataset
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    problems = []
    correct_answers = []
    
    for item in dataset:
        question = item["Question"]
        # GPQA has Correct Answer and Incorrect Answer 1-3
        correct_answer = item["Correct Answer"]
        choices = [
            correct_answer,
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        
        # Shuffle choices to randomize position of correct answer
        random.shuffle(choices)
        
        # Find the index of correct answer after shuffling
        correct_index = choices.index(correct_answer)
        
        # Format as multiple choice question
        input_prompt = f"{question.strip()}\nA. {choices[0].strip()}\nB. {choices[1].strip()}\nC. {choices[2].strip()}\nD. {choices[3].strip()}\nAnswer:"
        
        problems.append(input_prompt)
        correct_answers.append(multiple_choice_num_to_letter(correct_index))
    
    print(f"Evaluating on {len(problems)} GPQA-Diamond problems...")
    
    # Generate responses
    formatted_problems, responses = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=problems,
        apply_chat_template=apply_chat_template,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )
    
    # Evaluate
    correct = 0
    results = []
    
    for i, (response, correct_answer) in enumerate(zip(responses, correct_answers)):
        is_correct = grade_answer(response, correct_answer, extract=True, added_answer_extraction=True)
        extracted_answer = extract_last_boxed(response, added_answer_extraction=True)
        if is_correct:
            correct += 1
        
        results.append({
            "problem": problems[i],
            "formatted_problem": formatted_problems[i],
            "ground_truth": correct_answer,
            "raw_response": response,
            "extracted_answer": extracted_answer.strip(), 
            "correct": is_correct,
        })
    
    accuracy = correct / len(results) if results else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "samples": results[:log_n_results],
    }
