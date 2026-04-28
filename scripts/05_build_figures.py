import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_theme(style="whitegrid")

def build_bar_with_error(df, metric, output_path):
    plt.figure(figsize=(10, 6))
    means = df[f'{metric}_mean']
    stds = df[f'{metric}_std']
    groups = df['group']
    
    plt.bar(groups, means, yerr=stds, capsize=10, color=['skyblue', 'lightgreen', 'salmon', 'orchid'])
    plt.title(f'FD001 {metric.upper()} (Mean ± Std) by Group')
    plt.ylabel(metric.upper())
    plt.xlabel('Experimental Group')
    plt.savefig(output_path)
    plt.close()

def build_mean_variance_plot(df, metric, output_path):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[f'{metric}_mean'], df[f'{metric}_std'], s=100, color='red')
    
    for i, txt in enumerate(df['group']):
        plt.annotate(txt, (df[f'{metric}_mean'].iloc[i], df[f'{metric}_std'].iloc[i]), 
                     xytext=(5, 5), textcoords='offset points')
        
    plt.title(f'FD001 {metric.upper()} Mean-Variance Trade-off')
    plt.xlabel(f'Mean {metric.upper()}')
    plt.ylabel(f'Std Dev {metric.upper()}')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def build_g4_variability_boxplot(detail_df, output_path):
    if detail_df is None:
        return
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(y='rmse', data=detail_df, color='orchid')
    plt.title('G4 (LinEx + MCD) RMSE Variability Across Seeds')
    plt.ylabel('RMSE')
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/canonical_summary/group_comparison.csv")
    parser.add_argument("--detail-g4", type=str, default="results/canonical_summary/G4_multiseed_detail.csv")
    parser.add_argument("--output-dir", type=str, default="docs/figures")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
        
    df = pd.read_csv(args.input)
    
    # Sort groups
    df['group_idx'] = df['group'].str.extract('(\d+)').astype(int)
    df = df.sort_values('group_idx')
    
    print("Generating bar plots with error bars...")
    build_bar_with_error(df, 'rmse', out_dir / 'fd001_rmse_comparison.png')
    build_bar_with_error(df, 'mae', out_dir / 'fd001_mae_comparison.png')
    build_bar_with_error(df, 'nasa', out_dir / 'fd001_nasa_score_comparison.png')
    
    print("Generating mean-variance plots...")
    build_mean_variance_plot(df, 'rmse', out_dir / 'fd001_mean_variance_tradeoff_rmse.png')
    build_mean_variance_plot(df, 'nasa', out_dir / 'fd001_mean_variance_tradeoff_nasa.png')
    
    if os.path.exists(args.detail_g4):
        print("Generating G4 variability boxplot...")
        detail_df = pd.read_csv(args.detail_g4)
        build_g4_variability_boxplot(detail_df, out_dir / 'fd001_g4_rmse_boxplot.png')
    
    print(f"Figures saved to {out_dir}")

if __name__ == "__main__":
    main()
