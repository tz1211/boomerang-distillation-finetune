"""Utility functions for task evaluation."""

import re
from typing import List, Optional

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def extract_number(text: str) -> Optional[float]:
    """Extract the last number from text, handling various formats."""
    # Remove commas and other formatting
    text = text.replace(",", "")
    
    # Try to find numbers in various formats
    patterns = [
        r"-?\d+\.?\d*",  # Standard number
        r"\$(\d+\.?\d*)",  # Dollar amounts
        r"(\d+\.?\d*)\s*%",  # Percentages
    ]
    
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            if isinstance(matches[0], tuple):
                numbers.extend([float(m[0]) if m[0] else None for m in matches])
            else:
                numbers.extend([float(m) for m in matches])
    
    if numbers:
        return numbers[-1]  # Return the last number found
    
    return None


def extract_answer_boxed(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} LaTeX format."""
    # Look for \boxed{...}
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1].strip()
    
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