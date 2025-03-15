# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

#utils/memory_utils.py
import psutil
import os
import logging
from pathlib import Path
import torch
import gc

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor memory usage during training"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            'ram_used_gb': mem_info.rss / (1024 ** 3),  # GB
            'ram_percent': process.memory_percent(),
            'gpu_used_gb': torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0,
            'gpu_cached_gb': torch.cuda.memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0
        }
    
    @staticmethod
    def log_memory(step: int = None, prefix: str = ''):
        """Log current memory usage"""
        mem_stats = MemoryMonitor.get_memory_usage()
        
        msg = [f"{prefix}Memory usage:"]
        msg.append(f"RAM: {mem_stats['ram_used_gb']:.2f}GB ({mem_stats['ram_percent']:.1f}%)")
        
        if torch.cuda.is_available():
            msg.append(f"GPU: {mem_stats['gpu_used_gb']:.2f}GB used, "
                      f"{mem_stats['gpu_cached_gb']:.2f}GB cached")
            
        logger.info(' '.join(msg))
        return mem_stats

    @staticmethod
    def clear_memory():
        """Attempt to clear unused memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
