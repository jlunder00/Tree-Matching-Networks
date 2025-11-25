#!/usr/bin/env python3
"""
Generate publication-quality training plots from downloaded WandB data.

This script creates:
1. Training/validation loss curves
2. Contrastive metric curves (pos vs neg for similarity, distance, midpoint)
3. Gap evolution curves
4. Cross-model comparison plots

Usage:
    python generate_training_plots.py --model-family tmn_embedding_snli --stage primary
    python generate_training_plots.py --all
    python generate_training_plots.py --comparisons
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import seaborn as sns

# Set publication-quality defaults
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 13

# Color palette
COLORS = {
    'train': '#2E86AB',  # Blue
    'val': '#A23B72',    # Purple
    'pos': '#06A77D',    # Green
    'neg': '#D64933',    # Red
    'gap': '#F18F01',    # Orange
}

RESULTS_BASE = Path('/home/jlunder/research/results')


def load_history(model_family: str, stage: str) -> pd.DataFrame:
    """Load training history CSV for a given model family and stage."""
    history_file = RESULTS_BASE / model_family / stage / 'history.csv'
    if not history_file.exists():
        raise FileNotFoundError(f"History file not found: {history_file}")
    return pd.read_csv(history_file)


def plot_loss_curves(history: pd.DataFrame, model_family: str, stage: str, save_dir: Path):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Extract loss data
    train_loss = history['train.loss'].dropna()
    val_loss = history['val.loss'].dropna()

    # Plot with confidence intervals if multiple values per epoch
    epochs_train = history[history['train.loss'].notna()]['epoch'].values
    epochs_val = history[history['val.loss'].notna()]['epoch'].values

    ax.plot(epochs_train, train_loss, label='Training Loss',
            color=COLORS['train'], linewidth=2, alpha=0.9)
    ax.plot(epochs_val, val_loss, label='Validation Loss',
            color=COLORS['val'], linewidth=2, alpha=0.9)

    # Formatting
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_family.upper()} - {stage.capitalize()} Stage: Training Loss')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Save
    output_file = save_dir / 'loss_curves.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved loss curves to {output_file}")


def plot_contrastive_metrics(history: pd.DataFrame, model_family: str, stage: str, save_dir: Path):
    """Plot contrastive metrics: positive vs negative for similarity, distance, midpoint."""

    # Check which metrics are available
    available_metrics = []
    metric_pairs = [
        ('similarity', 'train.pos_similarity', 'train.neg_similarity'),
        ('distance', 'train.pos_distance', 'train.neg_distance'),
        ('midpoint', 'train.pos_midpoint', 'train.neg_midpoint'),
    ]

    for metric_name, pos_col, neg_col in metric_pairs:
        if pos_col in history.columns and neg_col in history.columns:
            available_metrics.append((metric_name, pos_col, neg_col))

    if not available_metrics:
        print(f"  ⚠ No contrastive metrics found in history")
        return

    # Create subplots
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics))
    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_name, pos_col, neg_col) in zip(axes, available_metrics):
        pos_data = history[pos_col].dropna()
        neg_data = history[neg_col].dropna()

        epochs_pos = history[history[pos_col].notna()]['epoch'].values
        epochs_neg = history[history[neg_col].notna()]['epoch'].values

        ax.plot(epochs_pos, pos_data, label=f'Positive {metric_name.capitalize()}',
                color=COLORS['pos'], linewidth=2, alpha=0.9)
        ax.plot(epochs_neg, neg_data, label=f'Negative {metric_name.capitalize()}',
                color=COLORS['neg'], linewidth=2, alpha=0.9)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f'{metric_name.capitalize()}: Positive vs Negative')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f'{model_family.upper()} - {stage.capitalize()} Stage: Contrastive Metrics',
                 y=1.00, fontsize=14)
    plt.tight_layout()

    output_file = save_dir / 'contrastive_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved contrastive metrics to {output_file}")


def plot_gap_evolution(history: pd.DataFrame, model_family: str, stage: str, save_dir: Path):
    """Plot gap evolution (similarity_gap, distance_gap, midpoint_gap)."""

    # Check which gaps are available
    gap_cols = []
    gap_names = []
    for col in history.columns:
        if 'gap' in col.lower() and 'batch/' in col:
            gap_cols.append(col)
            gap_names.append(col.split('/')[-1].replace('_', ' ').title())

    if not gap_cols:
        print(f"  ⚠ No gap metrics found in history")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for gap_col, gap_name in zip(gap_cols, gap_names):
        gap_data = history[gap_col].dropna()
        epochs = history[history[gap_col].notna()]['epoch'].values
        ax.plot(epochs, gap_data, label=gap_name, linewidth=2, alpha=0.9)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap (Positive - Negative)')
    ax.set_title(f'{model_family.upper()} - {stage.capitalize()} Stage: Gap Evolution')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    output_file = save_dir / 'gap_evolution.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved gap evolution to {output_file}")


def generate_plots_for_run(model_family: str, stage: str):
    """Generate all plots for a single run."""
    print(f"\n{'='*80}")
    print(f"Generating plots for {model_family} - {stage}")
    print(f"{'='*80}")

    # Load history
    try:
        history = load_history(model_family, stage)
        print(f"  Loaded history: {len(history)} samples, {len(history.columns)} columns")
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        return

    save_dir = RESULTS_BASE / model_family / stage
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    try:
        plot_loss_curves(history, model_family, stage, save_dir)
    except Exception as e:
        print(f"  ✗ Error generating loss curves: {e}")

    try:
        plot_contrastive_metrics(history, model_family, stage, save_dir)
    except Exception as e:
        print(f"  ✗ Error generating contrastive metrics: {e}")

    try:
        plot_gap_evolution(history, model_family, stage, save_dir)
    except Exception as e:
        print(f"  ✗ Error generating gap evolution: {e}")

    print(f"\n✓ Completed plots for {model_family} - {stage}")


def generate_comparison_plots():
    """Generate cross-model comparison plots."""
    print(f"\n{'='*80}")
    print("Generating comparison plots...")
    print(f"{'='*80}")

    # TODO: Implement comparison plots
    # - Compare loss curves across models for same dataset/stage
    # - Compare convergence speed
    # - Compare final metrics

    print("  ⚠ Comparison plots not yet implemented")


def main():
    parser = argparse.ArgumentParser(description='Generate training plots from WandB data')
    parser.add_argument('--model-family', type=str, help='Model family (e.g., tmn_embedding_snli)')
    parser.add_argument('--stage', type=str, help='Training stage (e.g., primary, finetune)')
    parser.add_argument('--all', action='store_true', help='Generate plots for all downloaded data')
    parser.add_argument('--comparisons', action='store_true', help='Generate comparison plots')

    args = parser.parse_args()

    if args.all:
        # Find all directories with history.csv
        for model_dir in RESULTS_BASE.iterdir():
            if model_dir.is_dir():
                for stage_dir in model_dir.iterdir():
                    if stage_dir.is_dir() and (stage_dir / 'history.csv').exists():
                        generate_plots_for_run(model_dir.name, stage_dir.name)

    elif args.comparisons:
        generate_comparison_plots()

    elif args.model_family and args.stage:
        generate_plots_for_run(args.model_family, args.stage)

    else:
        print("Error: Must specify --model-family and --stage, --all, or --comparisons")
        print("\nExample:")
        print("  python generate_training_plots.py --model-family tmn_embedding_snli --stage primary")
        print("  python generate_training_plots.py --all")


if __name__ == '__main__':
    main()
