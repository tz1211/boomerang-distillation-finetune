"""
Custom evaluation framework without lm_eval_harness.
Supports: AIME, GPQA, GSM8K, IF-EVAL, MATH500, MMLU
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
from typing import List, Optional, Union

import numpy as np
import torch
from vllm import LLM

from patching.patch import build_intermediate_model
from evaluate.system_prompts import SYSTEM_PROMPTS
from evaluate.tasks.gsm8k import evaluate_gsm8k
from evaluate.tasks.math500 import evaluate_math500
from evaluate.tasks.aime import evaluate_aime
from evaluate.tasks.mmlu import evaluate_mmlu
from evaluate.tasks.gpqa import evaluate_gpqa
from evaluate.tasks.ifeval import evaluate_ifeval


def to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    if isinstance(obj, np.dtype):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_n_params(model):
    """
    Get the number of training-time and inference-time parameters of the model.
    
    Args:
        model: The model to be evaluated.

    Returns:
        A dictionary with 'training_time' and 'inference_time' parameter counts.
    """
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters()) if hasattr(model, "lm_head") else 0
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "training_time": total_params - lm_head_params,
        "inference_time": total_params,
    }


TASK_EVALUATORS = {
    "gsm8k": evaluate_gsm8k,
    "math500": evaluate_math500,
    "aime": evaluate_aime,
    "mmlu": evaluate_mmlu,
    "gpqa": evaluate_gpqa,
    "ifeval": evaluate_ifeval,
}


def main(
    teacher_name_or_path: str,
    student_name_or_path: str,
    save_directory: str,
    tasks: Union[str, List[str]],
    num_layers_to_patch: int = 1,
    patch_first_k_layers: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    apply_chat_template: bool = True,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    limit: Optional[int] = None,
    vllm_tensor_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.9,
    use_system_prompts: bool = True,
    log_n_results: int = 0,
) -> str:
    """
    Main evaluation function using vLLM for inference.
    
    Args:
        teacher_name_or_path: Path to teacher model
        student_name_or_path: Path to student model
        save_directory: Directory to save results
        tasks: Task name(s) to evaluate on
        num_layers_to_patch: Number of layers to patch
        patch_first_k_layers: Whether to patch first k layers
        dtype: Model dtype
        apply_chat_template: Whether to apply chat template
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        limit: Limit number of examples per task
        vllm_tensor_parallel_size: Tensor parallelism size for vLLM
        vllm_gpu_memory_utilization: GPU memory utilization for vLLM
        use_system_prompts: Whether to use system prompts from SYSTEM_PROMPTS dictionary
        log_n_results: Number of results to log/print to console for each task (0 to disable)
        
    Returns:
        Path to saved results file
    """
    model_results = {}
    temp_model_dir = None
    
    # Build intermediate model
    print("Building intermediate model...")
    intermediate, tokenizer = build_intermediate_model(
        teacher_name_or_path=teacher_name_or_path,
        student_name_or_path=student_name_or_path,
        num_layers_to_patch=num_layers_to_patch,
        patch_first_k_layers=patch_first_k_layers,
        dtype=dtype,
    )
    
    print(f"Final number of layers: {intermediate.config.num_hidden_layers}")
    model_results["parameters"] = get_n_params(intermediate)
    model_results["num_layers"] = intermediate.config.num_hidden_layers
    
    # Save and load model with vLLM
    print("Saving intermediate model temporarily for vLLM...")
    temp_model_dir = tempfile.mkdtemp(prefix="vllm_eval_model_")
    intermediate.save_pretrained(temp_model_dir)
    tokenizer.save_pretrained(temp_model_dir)
    
    # Clean up intermediate model from memory
    del intermediate
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Loading model with vLLM...")
    llm = LLM(
        model=temp_model_dir,
        tensor_parallel_size=vllm_tensor_parallel_size,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        dtype=str(dtype).replace("torch.", ""),
        trust_remote_code=True,
    )
    print("vLLM model loaded successfully!")
    
    # Parse tasks
    if isinstance(tasks, str):
        task_list = [t.strip() for t in tasks.split(",")]
    else:
        task_list = tasks
    
    # Evaluate on each task
    all_results = {}
    
    for task_name in task_list:
        if task_name not in TASK_EVALUATORS:
            print(f"Warning: Unknown task '{task_name}', skipping...")
            continue
        
        print(f"\nEvaluating on {task_name}...")
        evaluator_fn = TASK_EVALUATORS[task_name]
        
        try:
            # Get system prompt for this task if enabled
            task_system_prompt = SYSTEM_PROMPTS.get(task_name, None) if use_system_prompts else None
            
            task_results = evaluator_fn(
                model=llm,
                tokenizer=tokenizer,
                apply_chat_template=apply_chat_template,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                limit=limit,
                system_prompt=task_system_prompt,
                log_n_results=log_n_results,
            )
            all_results[task_name] = task_results
            print(f"{task_name} results: {task_results.get('accuracy', task_results.get('score', 'N/A'))}")
        except Exception as e:
            print(f"Error evaluating {task_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task_name] = {"error": str(e)}
    
    model_results["results"] = all_results
    
    # Save results
    tasks_str = "_".join(task_list) if task_list else "all"
    # Get num_layers from model_results since intermediate is deleted
    num_layers = model_results["num_layers"]
    curr_out_file = os.path.join(
        save_directory,
        f"custom_eval_patch_first_{patch_first_k_layers}_n_layers_{num_layers}_{tasks_str}.json",
    )
    
    os.makedirs(save_directory, exist_ok=True)
    
    with open(curr_out_file, "w") as f:
        json.dump(model_results, f, default=to_serializable, indent=2)
    
    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # Clean up temporary model directory
    if temp_model_dir and os.path.exists(temp_model_dir):
        print(f"Cleaning up temporary model directory: {temp_model_dir}")
        shutil.rmtree(temp_model_dir)
    
    return curr_out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom evaluation framework")
    parser.add_argument(
        "--teacher_model_name_or_path",
        required=True,
        help="Huggingface model name or local path for the teacher model",
    )
    parser.add_argument(
        "--student_model_name_or_path",
        required=True,
        help="Student model name or path",
    )
    parser.add_argument(
        "--save_directory",
        required=True,
        help="Directory to save the results in",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="Comma-separated list of tasks: aime,gpqa,gsm8k,ifeval,math500,mmlu",
    )
    parser.add_argument(
        "--num_layers_to_patch",
        type=int,
        default=1,
        help="Number of student layers to patch",
    )
    parser.add_argument(
        "--patch_first_k_layers",
        action="store_true",
        help="Patch student layers starting from first layers if true, last layers otherwise",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--apply_chat_template",
        type=bool,
        default=True,
        required=False,
        help="Apply chat template to prompts",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy decoding)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task",
    )
    parser.add_argument(
        "--override_llama_patching",
        action="store_true",
        help="If true, overrides the default patching order behavior for llama",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallelism size for vLLM (default: 1)",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (default: 0.9)",
    )
    parser.add_argument(
        "--use_system_prompts",
        type=bool,
        default=True,
        required=False,
        help="Disable system prompts (system prompts are enabled by default from evaluate.system_prompts.SYSTEM_PROMPTS)",
    )
    parser.add_argument(
        "--log_n_results",
        type=int,
        default=0,
        help="Number of results to log for each task (0 to disable, default: 0)",
    )
    
    args = parser.parse_args()
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    
    if (
        not args.override_llama_patching
        and "llama" in args.teacher_model_name_or_path.lower()
    ):
        print("Using default patching order for llama models (first k layers)")
        args.patch_first_k_layers = True
    
    out_file = main(
        teacher_name_or_path=args.teacher_model_name_or_path,
        student_name_or_path=args.student_model_name_or_path,
        save_directory=args.save_directory,
        tasks=args.tasks,
        num_layers_to_patch=args.num_layers_to_patch,
        patch_first_k_layers=args.patch_first_k_layers,
        dtype=dtype_map[args.dtype],
        apply_chat_template=args.apply_chat_template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        limit=args.limit,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        use_system_prompts=args.use_system_prompts,
        log_n_results=args.log_n_results,
    )
    
    print(f"\nResults saved to {out_file}")

