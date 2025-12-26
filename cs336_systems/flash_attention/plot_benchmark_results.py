import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_benchmark(csv_file, output_dir=None):
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return

    # If output_dir is not specified, save in the same directory as the CSV
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_file))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(csv_file)
    
    # Filter out rows with NaN in 'FlashAttn_fwd' (failed runs) for the plot to be clean
    # or keep them to show gaps. Let's filter for now but warn.
    if df['FlashAttn_fwd'].isna().any():
        print("Warning: Some FlashAttention runs failed (NaN contents). Comparisons may be incomplete.")
    
    # Setup plot style
    sns.set_style("whitegrid")
    
    # Get unique dimensions and dtypes
    dims = sorted(df['dim'].unique())
    dtypes = sorted(df['dtype'].unique())
    
    for dtype in dtypes:
        subset_dtype = df[df['dtype'] == dtype]
        if subset_dtype.empty:
            continue
            
        n_dims = len(dims)
        fig, axes = plt.subplots(1, n_dims, figsize=(n_dims * 5, 5), sharey=True)
        if n_dims == 1: axes = [axes]
        
        fig.suptitle(f"Benchmark Results - {dtype}", fontsize=16)
        
        for i, dim in enumerate(dims):
            ax = axes[i]
            data = subset_dtype[subset_dtype['dim'] == dim]
            
            # Melt for easier plotting with seaborn
            plot_df = data.melt(
                id_vars=['seq_len'], 
                value_vars=['Torch(Eager)_fwd', 'Torch(Compile)_fwd', 'FlashAttn_fwd'],
                var_name='Implementation', 
                value_name='Latency (ms)'
            )
            
            sns.lineplot(data=plot_df, x='seq_len', y='Latency (ms)', hue='Implementation', marker='o', ax=ax)
            
            ax.set_title(f"Head Dim = {dim}")
            ax.set_xlabel("Sequence Length")
            if i == 0:
                ax.set_ylabel("Forward Latency (ms)")
            else:
                ax.set_ylabel("")
            
            ax.set_xscale('log', base=2)
            ax.set_yscale('log') # Log scale for latency often helps see differences
            
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"benchmark_plot_{dtype}.png")
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results from CSV")
    parser.add_argument("csv_file", type=str, help="Path to the benchmark results CSV file")
    args = parser.parse_args()
    
    plot_benchmark(args.csv_file)
