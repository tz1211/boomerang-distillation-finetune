"""Utility functions for task evaluation."""

import re
from typing import List, Optional

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def extract_number(text: str) -> Optional[float]:
    """Extract the last number from text, handling various formats."""
    if text is None:
        return None
    
    text = text.replace(",", "")
    
    # Find all numbers (handles negatives and decimals)
    matches = re.findall(r"-?\d+\.?\d*", text)
    
    if matches:
        return float(matches[-1])  # Return the last number found
    
    return None

def extract_answer_boxed(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} LaTeX format, handling nested braces."""
    # Find all occurrences of \boxed{
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return None
    
    # Start from the last match (most likely the final answer)
    match = matches[-1]
    start_pos = match.end()  # Position after \boxed{
    
    # Count braces to find the matching closing brace
    brace_count = 1  # We already have one opening brace from \boxed{
    pos = start_pos
    
    while pos < len(text) and brace_count > 0:
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
        pos += 1
    
    if brace_count == 0:
        # Found matching closing brace
        return text[start_pos:pos-1].strip()  # pos-1 to exclude the closing brace
    
    return None

def extract_final_answer(text: str) -> Optional[str]:
    """Extract final answer from text, trying multiple strategies."""
    # Try boxed format first
    boxed = extract_answer_boxed(text)
    if boxed:
        return boxed
    
    # Look for "The answer is" or similar patterns
    patterns = [
        r"(?:The |Final )?answer is[:\s]+([^\n\.]+)",
        r"(?:Therefore|Thus|So)[,\s]+(?:the )?answer is[:\s]+([^\n\.]+)",
        r"Answer[:\s]+([^\n\.]+)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    
    # Try to extract the last number
    number = extract_number(text)
    if number is not None:
        return str(number)
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove whitespace
    answer = answer.strip()
    
    # Remove common prefixes
    answer = re.sub(r"^(the answer is|answer:|answer is|final answer:?)\s*", "", answer, flags=re.IGNORECASE)
    
    # Remove dollar signs and commas
    answer = answer.replace("$", "").replace(",", "")
    
    # Remove LaTeX formatting
    answer = re.sub(r"\\[a-zA-Z]+\{([^}]+)\}", r"\1", answer)
    answer = answer.replace("$", "").replace("\\", "")
    
    # Remove extra whitespace
    answer = " ".join(answer.split())
    
    return answer.lower()


def batch_generate(
    model: LLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    apply_chat_template: bool = True,
    system_prompt: Optional[str] = None,
) -> List[str]:
    """
    Generate responses for a batch of prompts using vLLM.
    
    Args:
        model: The vLLM model
        tokenizer: The tokenizer
        prompts: List of input prompts
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        apply_chat_template: Whether to apply chat template
        system_prompt: Optional system prompt to prepend to each user message
        
    Returns:
        List of generated text responses
    """
    # Optionally format prompts using chat template
    has_chat = (
        apply_chat_template
        and getattr(tokenizer, "chat_template", None)
        and hasattr(tokenizer, "apply_chat_template")
    )
    if apply_chat_template and not has_chat:
        raise ValueError("apply_chat_template=True but tokenizer is missing a valid chat_template or apply_chat_template method.")

    if has_chat:
        formatted_prompts = []
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted)
    else:
        # If no chat template, prepend system prompt directly if provided
        if system_prompt:
            formatted_prompts = [f"{system_prompt}\n\n{prompt}" for prompt in prompts]
        else:
            formatted_prompts = prompts
    
    # Create sampling params
    sampling_params = SamplingParams(
        temperature=temperature if temperature > 0 else 0.0,
        max_tokens=max_new_tokens,
    )
    
    # Generate
    outputs = model.generate(formatted_prompts, sampling_params)
    
    # Extract responses
    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text
        responses.append(generated_text.strip())
    
    return formatted_prompts, responses