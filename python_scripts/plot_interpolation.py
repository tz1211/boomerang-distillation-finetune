import json
import os
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Supported benchmarks in the new custom evaluation pipeline
SUPPORTED_BENCHMARKS = [
    "gsm8k",
    "math500",
    "aime",
    "mmlu",
    "gpqa",
    "livecodebench",
]


def plot_interpolation(
    benchmark="gsm8k",
    base_dir="results/eval",
    model_variant="Qwen",
    output_dir="figs",
):
    """
    Plot interpolation results for different models on a given benchmark.
    
    Args:
        benchmark: Name of the benchmark (e.g., "gsm8k", "math500", "aime", "mmlu", "gpqa", "livecodebench")
        base_dir: Base directory containing evaluation results (default: "results/eval")
        model_variant: Model variant directory name (default: "Qwen")
        output_dir: Directory to save the output plot (default: "figs")
    """
    if benchmark not in SUPPORTED_BENCHMARKS:
        print(f"Warning: Benchmark '{benchmark}' not in supported list: {SUPPORTED_BENCHMARKS}")
        print(f"Attempting to proceed anyway...")
    
    # Construct the benchmark directory path
    benchmark_dir = os.path.join(base_dir, benchmark, model_variant)
    
    if not os.path.exists(benchmark_dir):
        print(f"Error: Directory not found: {benchmark_dir}")
        return
    
    # Find all model directories (e.g., Qwen3-4B, Qwen3-4B-Base, etc.)
    model_dirs = [
        d for d in os.listdir(benchmark_dir)
        if os.path.isdir(os.path.join(benchmark_dir, d))
    ]
    
    if not model_dirs:
        print(f"No model directories found in {benchmark_dir}")
        return
    
    # Store data for each model
    interpolation_data = {}
    
    for model_name in sorted(model_dirs):
        model_path = os.path.join(benchmark_dir, model_name)
        
        # Find all JSON files matching the custom eval pattern
        json_files = glob.glob(os.path.join(model_path, f"custom_eval_*_{benchmark}.json"))
        
        if not json_files:
            print(f"Warning: No result files found for {model_name} in {model_path}")
            continue
        
        # Collect interpolation data
        params_list = []
        accuracy_list = []
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    result = json.load(f)
                
                # Extract number of parameters (inference_time)
                num_params = result.get("parameters", {}).get("inference_time", 0)
                
                # Extract accuracy from results
                task_results = result.get("results", {}).get(benchmark, {})
                
                # Check if there's an error
                if "error" in task_results:
                    print(f"Warning: Error in {os.path.basename(json_file)}: {task_results['error']}")
                    continue
                
                accuracy = task_results.get("accuracy")
                
                # Only add if we have valid data
                if num_params > 0 and accuracy is not None:
                    params_list.append(num_params)
                    accuracy_list.append(accuracy)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                continue
        
        # Sort by number of parameters
        if params_list:  # Only add if we have data
            sorted_data = sorted(zip(params_list, accuracy_list))
            interpolation_data[model_name] = {
                "params": [p for p, a in sorted_data],
                "accuracy": [a for p, a in sorted_data]
            }
    
    if not interpolation_data:
        print(f"No data found for benchmark '{benchmark}'")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Get color cycle to ensure matching colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Plot interpolation lines for all models
    for idx, (model_name, model_data) in enumerate(interpolation_data.items()):
        params = np.array(model_data["params"])
        accuracy = np.array(model_data["accuracy"])
        
        # Convert params to billions for readability
        params_billions = params / 1e9
        
        # Use color from cycle, cycling if needed
        color = colors[idx % len(colors)]
        
        plt.plot(params_billions, accuracy, marker='o', label=model_name, 
                linewidth=2, markersize=8, color=color)
    
    # Format benchmark name for display
    benchmark_display = benchmark.upper().replace("_", " ")
    
    plt.xlabel("Number of Parameters (Billions)", fontsize=12)
    plt.ylabel(f"{benchmark_display} Accuracy", fontsize=12)
    plt.title(f"Interpolation Results: {benchmark_display} Accuracy vs Model Size", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"interpolation_results_{benchmark}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # Also display the plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot interpolation results for different benchmarks using the custom evaluation pipeline"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="gsm8k",
        help=f"Benchmark name. Supported: {', '.join(SUPPORTED_BENCHMARKS)}. Default: 'gsm8k'"
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
        "--output_dir",
        type=str,
        default="figs/interpolation_eval",
        help="Directory to save output plots. Default: 'figs'"
    )
    
    args = parser.parse_args()
    plot_interpolation(
        benchmark=args.benchmark,
        base_dir=args.base_dir,
        model_variant=args.model_variant,
        output_dir=args.output_dir,
    )
