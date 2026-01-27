#!/usr/bin/env python3
"""
Download WandB data (metrics, history, graphics) for all completed runs.

Usage:
    python download_wandb_data.py [--run-ids RUN_ID1,RUN_ID2,...] [--all]

Examples:
    # Download data for specific runs
    python download_wandb_data.py --run-ids vn9wc4t0,40v7x052

    # Download all completed TMN runs
    python download_wandb_data.py --all
"""

import wandb
import json
import os
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

# Define all completed runs with their metadata
COMPLETED_RUNS = {
    # TMN Matching
    'vn9wc4t0': {
        'model_family': 'tmn_matching',
        'dataset': 'wikiqs',
        'stage': 'contrastive',
        'description': 'TMN Matching Contrastive Pretraining'
    },
    '40v7x052': {
        'model_family': 'tmn_matching_snli',
        'dataset': 'snli',
        'stage': 'primary',
        'description': 'TMN Matching SNLI Primary'
    },
    # TMN Embedding
    'f585gcqh': {
        'model_family': 'tmn_embedding',
        'dataset': 'wikiqs',
        'stage': 'contrastive',
        'description': 'TMN Embedding Contrastive Pretraining'
    },
    '3cbyh3ih': {
        'model_family': 'tmn_embedding_semeval',
        'dataset': 'semeval',
        'stage': 'primary',
        'description': 'TMN Embedding SemEval Primary'
    },
    'zkvpelnb': {
        'model_family': 'tmn_embedding_semeval',
        'dataset': 'semeval',
        'stage': 'finetune',
        'description': 'TMN Embedding SemEval Finetune'
    },
    'o2pk1gzl': {
        'model_family': 'tmn_embedding_semeval',
        'dataset': 'semeval',
        'stage': 'evaluation',
        'description': 'TMN Embedding SemEval Evaluation'
    },
    '0qikr4db': {
        'model_family': 'tmn_embedding_snli',
        'dataset': 'snli',
        'stage': 'primary',
        'description': 'TMN Embedding SNLI Primary'
    },
    '7lfnzo1s': {
        'model_family': 'tmn_embedding_snli',
        'dataset': 'snli',
        'stage': 'finetune',
        'description': 'TMN Embedding SNLI Finetune'
    },
    'wdneplwn': {
        'model_family': 'tmn_embedding_snli',
        'dataset': 'snli',
        'stage': 'evaluation',
        'description': 'TMN Embedding SNLI Evaluation'
    },
}

# Base results directory
RESULTS_BASE = Path('/home/jlunder/research/results')


def download_run_data(run_id: str, metadata: Dict, api: wandb.Api, verbose: bool = True):
    """Download all data for a single run."""

    if verbose:
        print(f"\n{'='*80}")
        print(f"Downloading: {metadata['description']} ({run_id})")
        print(f"{'='*80}")

    # Get run from WandB
    run = api.run(f'jlunder00/tree-embedding/{run_id}')

    # Create results directory
    results_dir = RESULTS_BASE / metadata['model_family'] / metadata['stage']
    results_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Results directory: {results_dir}")

    # 1. Download summary metrics
    if verbose:
        print("\n[1/4] Downloading summary metrics...")

    summary = dict(run.summary)
    # Remove internal wandb keys
    summary = {k: v for k, v in summary.items() if not k.startswith('_wandb')}

    metrics_file = results_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    if verbose:
        print(f"  Saved {len(summary)} metrics to {metrics_file}")

    # 2. Download history (training curves data)
    if verbose:
        print("\n[2/4] Downloading training history...")

    try:
        history = run.history(samples=100000)  # Get all available samples
        if len(history) > 0:
            history_file = results_dir / 'history.csv'
            history.to_csv(history_file, index=False)
            if verbose:
                print(f"   Saved {len(history)} samples, {len(history.columns)} columns to {history_file}")
                print(f"  Sample columns: {list(history.columns)[:10]}...")
        else:
            if verbose:
                print("   No history data available")
    except Exception as e:
        if verbose:
            print(f"   Error downloading history: {e}")

    # 3. Download media files (confusion matrices, scatter plots)
    if verbose:
        print("\n[3/4] Downloading media files...")

    media_count = 0
    try:
        files = run.files()
        for file in files:
            if any(ext in file.name for ext in ['.png', '.jpg', '.svg', '.pdf']):
                # Extract filename from path
                filename = file.name.split('/')[-1]
                output_path = results_dir / filename
                file.download(root=str(results_dir), replace=True)
                media_count += 1
                if verbose:
                    print(f"   Downloaded {filename}")

        if media_count == 0 and verbose:
            print("   No media files found")
    except Exception as e:
        if verbose:
            print(f"   Error downloading media: {e}")

    # 4. Save run configuration
    if verbose:
        print("\n[4/4] Saving run configuration...")

    config = dict(run.config)
    config_file = results_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    if verbose:
        print(f"   Saved configuration to {config_file}")

    if verbose:
        print(f"\n Completed download for {run_id}")
        print(f"  Total files: metrics.json, history.csv (if available), {media_count} media files, config.json")


def main():
    parser = argparse.ArgumentParser(description='Download WandB data for analysis')
    parser.add_argument('--run-ids', type=str, help='Comma-separated list of run IDs to download')
    parser.add_argument('--all', action='store_true', help='Download all completed runs')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')

    args = parser.parse_args()

    # Initialize WandB API
    api = wandb.Api()

    # Determine which runs to download
    if args.all:
        run_ids = list(COMPLETED_RUNS.keys())
        print(f"Downloading data for all {len(run_ids)} completed runs...")
    elif args.run_ids:
        run_ids = args.run_ids.split(',')
        print(f"Downloading data for {len(run_ids)} specified runs...")
    else:
        print("Error: Must specify --run-ids or --all")
        print("\nAvailable run IDs:")
        for run_id, metadata in COMPLETED_RUNS.items():
            print(f"  {run_id}: {metadata['description']}")
        return

    # Download data for each run
    success_count = 0
    for i, run_id in enumerate(run_ids, 1):
        if run_id not in COMPLETED_RUNS:
            print(f"\n Warning: Run ID {run_id} not found in COMPLETED_RUNS")
            continue

        try:
            metadata = COMPLETED_RUNS[run_id]
            print(f"\n[{i}/{len(run_ids)}] Processing {run_id}...")
            download_run_data(run_id, metadata, api, verbose=args.verbose)
            success_count += 1
        except Exception as e:
            print(f"\n Error processing {run_id}: {e}")

    print(f"\n{'='*80}")
    print(f"Download complete: {success_count}/{len(run_ids)} successful")
    print(f"Results saved to: {RESULTS_BASE}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
