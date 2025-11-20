"""
Evaluate completeness of model generations by analyzing token counts and accuracy.

For a given benchmark, plots graphs for thinking and finetuned models showing:
- Accuracy vs num_layers (left y-axis)
- Average token count or cut-off percentage vs num_layers (right y-axis)
"""

import json
import os
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from typing import Dict


def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    """Count the number of tokens in a text string."""
    if not text:
        return 0
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def load_model_data(
    benchmark: str,
    base_dir: str,
    model_variant: str,
    model_name: str,
    tokenizer: AutoTokenizer,
    use_cutoff: bool = False,
    max_tokens: int = 4096,
) -> Dict[int, Dict]:
    """
    Load evaluation data for a specific model.
    
    Returns:
        Dictionary mapping num_layers to {
            'accuracy': float,
            'avg_token_count': float,  # Average number of tokens per response
            'percent_cutoff': float,  # Percentage of responses cut-off (0-100)
            'total': int
        }
    """
    benchmark_dir = os.path.join(base_dir, benchmark, model_variant, model_name)
    
    if not os.path.exists(benchmark_dir):
        print(f"Warning: Directory not found: {benchmark_dir}")
        return {}
    
    # Find all JSON files matching the custom eval pattern
    json_files = glob.glob(os.path.join(benchmark_dir, f"custom_eval_*_{benchmark}.json"))
    
    if not json_files:
        print(f"Warning: No result files found for {model_name} in {benchmark_dir}")
        return {}
    
    data_by_layers = {}
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
            
            # Extract num_layers
            num_layers = result.get("num_layers")
            if num_layers is None:
                print(f"Warning: No num_layers found in {os.path.basename(json_file)}")
                continue
            
            # Extract accuracy from results
            task_results = result.get("results", {}).get(benchmark, {})
            
            # Check if there's an error
            if "error" in task_results:
                print(f"Warning: Error in {os.path.basename(json_file)}: {task_results['error']}")
                continue
            
            accuracy = task_results.get("accuracy")
            if accuracy is None:
                print(f"Warning: No accuracy found in {os.path.basename(json_file)}")
                continue
            
            # Count tokens for all responses and calculate metrics
            # If no top-level samples, check for by_subject (MMLU structure)
            if benchmark == "mmlu":
                by_subject = task_results.get("by_subject", {})
                if by_subject:
                    # Collect all samples from all subjects for overall statistics
                    samples = []
                    for subject_name, subject_data in by_subject.items():
                        if isinstance(subject_data, dict) and "samples" in subject_data:
                            samples.extend(subject_data["samples"])
            else: 
                samples = task_results.get("samples", [])
            
            total = len(samples)
            token_counts = []
            num_cutoff = 0
            
            for sample in samples:
                raw_response = sample.get("raw_response", "")
                if raw_response:
                    token_count = count_tokens(raw_response, tokenizer)
                    token_counts.append(token_count)
                    if token_count >= max_tokens:
                        num_cutoff += 1
            
            # Calculate average token count
            avg_token_count = np.mean(token_counts) if token_counts else 0.0
            # Calculate percentage of cut-off responses
            percent_cutoff = (num_cutoff / total * 100.0) if total > 0 else 0.0
            
            data_by_layers[num_layers] = {
                'accuracy': accuracy,
                'avg_token_count': avg_token_count,
                'percent_cutoff': percent_cutoff,
                'total': total
            }
            
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    return data_by_layers


def plot_completeness(
    benchmark: str,
    base_dir: str = "results/eval",
    model_variant: str = "Qwen",
    thinking_model: str = "Qwen3-4B-Thinking-2507",
    finetuned_model: str = "Qwen3-4B",
    output_dir: str = "figs",
    tokenizer_name: str = "Qwen/Qwen3-4B-Base",  # Use Qwen3 tokenizer to match Qwen3-4B models
    use_cutoff: bool = False,
    max_tokens: int = 4096,
):
    """
    Plot completeness analysis for thinking and finetuned models.
    
    Args:
        benchmark: Name of the benchmark (e.g., "gsm8k", "math500", "aime")
        base_dir: Base directory containing evaluation results
        model_variant: Model variant directory name
        thinking_model: Name of the thinking model directory
        finetuned_model: Name of the finetuned model directory
        output_dir: Directory to save output plots
        tokenizer_name: HuggingFace model name to load tokenizer from
        use_cutoff: If True, plot cut-off percentage; if False, plot average token count
        max_tokens: Token threshold for cut-off calculation (default: 4096)
    """
    print(f"Loading tokenizer from {tokenizer_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Trying alternative Qwen3 tokenizer...")
        try:
            # Try Qwen3-4B if Base variant fails
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
        except Exception as e2:
            print(f"Error loading alternative tokenizer: {e2}")
            return
    
    print(f"Loading data for thinking model: {thinking_model}")
    thinking_data = load_model_data(
        benchmark=benchmark,
        base_dir=base_dir,
        model_variant=model_variant,
        model_name=thinking_model,
        tokenizer=tokenizer,
        use_cutoff=use_cutoff,
        max_tokens=max_tokens,
    )
    
    print(f"Loading data for finetuned model: {finetuned_model}")
    finetuned_data = load_model_data(
        benchmark=benchmark,
        base_dir=base_dir,
        model_variant=model_variant,
        model_name=finetuned_model,
        tokenizer=tokenizer,
        use_cutoff=use_cutoff,
        max_tokens=max_tokens,
    )
    
    if not thinking_data and not finetuned_data:
        print(f"No data found for benchmark '{benchmark}'")
        return
    
    # Create plots for each model
    models_to_plot = []
    if thinking_data:
        models_to_plot.append(("Thinking", thinking_data, thinking_model))
    if finetuned_data:
        models_to_plot.append(("Finetuned", finetuned_data, finetuned_model))
    
    for model_type, data, model_name in models_to_plot:
        # Sort by num_layers
        sorted_layers = sorted(data.keys())
        layers = np.array(sorted_layers)
        accuracies = np.array([data[l]['accuracy'] for l in sorted_layers])
        
        # Choose metric based on mode
        if use_cutoff:
            y_values = np.array([data[l]['percent_cutoff'] for l in sorted_layers])
            y_label = f'Percentage of Responses Cut-off (≥{max_tokens} tokens)'
            y_legend = f'% Cut-off Responses (≥{max_tokens})'
            title_suffix = 'Cut-off Responses'
        else:
            y_values = np.array([data[l]['avg_token_count'] for l in sorted_layers])
            y_label = 'Average Token Count'
            y_legend = 'Average Token Count'
            title_suffix = 'Average Token Count'
        
        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot accuracy on left y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Number of Layers', fontsize=12)
        ax1.set_ylabel('Accuracy', color=color1, fontsize=12)
        line1 = ax1.plot(layers, accuracies, marker='o', color=color1, 
                        linewidth=2, markersize=8, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Plot metric on right y-axis
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel(y_label, color=color2, fontsize=12)
        line2 = ax2.plot(layers, y_values, marker='s', color=color2, 
                        linewidth=2, markersize=8, label=y_legend)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Format benchmark name for display
        benchmark_display = benchmark.upper().replace("_", " ")
        
        # Title
        plt.title(f'{model_type} Model: {benchmark_display}\n'
                 f'Accuracy and {title_suffix} vs Number of Layers', 
                 fontsize=14, pad=20)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        # Add mode suffix to filename to avoid overwriting
        mode_suffix = "cutoff" if use_cutoff else "tokens"
        output_path = os.path.join(
            output_dir, 
            f"completeness_{benchmark}_{model_name.replace('/', '_')}_{mode_suffix}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {output_path}")
        
        plt.close()
    
    print("Analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot completeness analysis (accuracy and average token count) vs number of layers"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Benchmark name (e.g., 'gsm8k', 'math500', 'aime')"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="results/eval",
        help="Base directory containing evaluation results. Default: 'results/eval'"
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="Qwen",
        help="Model variant directory name (e.g., 'Qwen'). Default: 'Qwen'"
    )
    parser.add_argument(
        "--thinking_model",
        type=str,
        default="Qwen3-4B-Thinking-2507",
        help="Name of the thinking model directory. Default: 'Qwen3-4B-Thinking-2507'"
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        default="Qwen3-4B",
        help="Name of the finetuned model directory. Default: 'Qwen3-4B'"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figs/completeness_eval",
        help="Directory to save output plots. Default: 'figs/completeness_eval'"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="Qwen/Qwen3-4B-Base",
        help="HuggingFace model name to load tokenizer from. Default: 'Qwen/Qwen3-4B-Base'"
    )
    parser.add_argument(
        "--use_cutoff",
        action="store_true",
        help="Plot cut-off percentage instead of average token count. Default: False (plot average token count)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Token threshold for cut-off calculation. Default: 4096"
    )
    
    args = parser.parse_args()
    plot_completeness(
        benchmark=args.benchmark,
        base_dir=args.base_dir,
        model_variant=args.model_variant,
        thinking_model=args.thinking_model,
        finetuned_model=args.finetuned_model,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        use_cutoff=args.use_cutoff,
        max_tokens=args.max_tokens,
    )

