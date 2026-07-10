"""Enhanced GPU utilities with nvitop integration for smart GPU selection"""
import os
import sys
import logging
from typing import List, Tuple

log = logging.getLogger(__name__)

# Try nvitop first, fallback to pynvml
try:
    from nvitop import Device
    NVITOP_AVAILABLE = True
except ImportError:
    NVITOP_AVAILABLE = False
    log.warning("nvitop not available, falling back to nvidia-ml-py")
    try:
        import pynvml
        PYNVML_AVAILABLE = True
    except ImportError:
        PYNVML_AVAILABLE = False
        log.warning("Neither nvitop nor pynvml available, using basic nvidia-smi")


def get_gpu_metrics_nvitop() -> List[Tuple[int, float, float, int]]:
    """Get GPU metrics using nvitop: (gpu_id, memory_percent, gpu_util_percent, memory_used_mb)"""
    devices = Device.all()
    gpu_metrics = []
    
    for device in devices:
        gpu_id = device.cuda_index
        memory_percent = device.memory_percent()
        gpu_util = device.gpu_utilization()
        memory_used_mb = int(device.memory_used() / 1024 / 1024) if device.memory_used() else 0
        gpu_metrics.append((gpu_id, memory_percent, gpu_util, memory_used_mb))
    
    return gpu_metrics


def get_gpu_metrics_pynvml() -> List[Tuple[int, float, float, int]]:
    """Fallback GPU metrics using pynvml"""
    if not PYNVML_AVAILABLE:
        return []
    
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    gpu_metrics = []
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used_mb = mem_info.used // 1024 // 1024
        memory_percent = (mem_info.used / mem_info.total) * 100
        
        # GPU utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
        except:
            gpu_util = 0  # fallback if utilization not available
        
        gpu_metrics.append((i, memory_percent, gpu_util, memory_used_mb))
    
    return gpu_metrics


def get_gpu_metrics_nvidia_smi() -> List[Tuple[int, float, float, int]]:
    """Basic GPU metrics using nvidia-smi"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, check=True)
        
        gpu_metrics = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                gpu_id = int(parts[0])
                used_mb = int(parts[1])
                total_mb = int(parts[2])
                gpu_util = int(parts[3]) if parts[3] != '[Not Supported]' else 0
                memory_percent = (used_mb / total_mb) * 100
                gpu_metrics.append((gpu_id, memory_percent, gpu_util, used_mb))
        
        return gpu_metrics
    except Exception as e:
        log.error(f"Failed to get GPU metrics via nvidia-smi: {e}")
        return []


def calculate_gpu_score(memory_percent: float, gpu_util: float, memory_weight: float = 0.7, util_weight: float = 0.3) -> float:
    """Calculate GPU suitability score (lower is better)"""
    return (memory_percent * memory_weight) + (gpu_util * util_weight)


def auto_select_gpus(num_gpus=1, min_free_memory_gb=2.0, memory_weight=0.7, util_weight=0.3):
    """
    Automatically select GPUs based on memory usage and utilization, set CUDA_VISIBLE_DEVICES.
    
    Args:
        num_gpus: Number of GPUs to select
        min_free_memory_gb: Minimum free memory required in GB
        memory_weight: Weight for memory usage in selection (default: 0.7)
        util_weight: Weight for GPU utilization in selection (default: 0.3)
        
    Returns:
        List of selected GPU IDs
    """
    try:
        # Get GPU metrics using the best available method
        if NVITOP_AVAILABLE:
            gpu_metrics = get_gpu_metrics_nvitop()
        elif PYNVML_AVAILABLE:
            gpu_metrics = get_gpu_metrics_pynvml()
        else:
            gpu_metrics = get_gpu_metrics_nvidia_smi()
        
        if not gpu_metrics:
            log.warning("No GPU metrics available, using default GPU selection")
            return list(range(num_gpus))
        
        # Calculate scores and filter by minimum free memory
        gpu_scores = []
        min_free_memory_mb = min_free_memory_gb * 1024
        
        for gpu_id, memory_percent, gpu_util, memory_used_mb in gpu_metrics:
            # Estimate total memory (this is approximate for nvidia-smi fallback)
            if NVITOP_AVAILABLE:
                device = Device(gpu_id)
                total_mb = device.memory_total() / 1024 / 1024
            else:
                total_mb = memory_used_mb / (memory_percent / 100) if memory_percent > 0 else 80000  # 80GB fallback
            
            free_mb = total_mb - memory_used_mb
            
            # Only consider GPUs with sufficient free memory
            if free_mb >= min_free_memory_mb:
                score = calculate_gpu_score(memory_percent, gpu_util, memory_weight, util_weight)
                gpu_scores.append((gpu_id, score, memory_percent, gpu_util, memory_used_mb, free_mb))
        
        if len(gpu_scores) < num_gpus:
            log.warning(f"Only {len(gpu_scores)} GPUs have >= {min_free_memory_gb}GB free memory, requested {num_gpus}")
            if len(gpu_scores) == 0:
                log.error("No GPUs meet memory requirements!")
                return list(range(num_gpus))  # Fallback to default
        
        # Sort by score (lower is better)
        gpu_scores.sort(key=lambda x: x[1])
        
        # Select the best GPUs
        selected_gpus = gpu_scores[:min(num_gpus, len(gpu_scores))]
        
        # Log selection info
        log.info("GPU Memory Status:")
        for gpu_id, memory_percent, gpu_util, memory_used_mb in gpu_metrics:
            # Calculate free memory for display
            if NVITOP_AVAILABLE:
                device = Device(gpu_id)
                total_mb = device.memory_total() / 1024 / 1024
            else:
                total_mb = memory_used_mb / (memory_percent / 100) if memory_percent > 0 else 80000
            
            free_mb = total_mb - memory_used_mb
            status = "SELECTED" if gpu_id in [g[0] for g in selected_gpus] else "        "
            log.info(f"  GPU {gpu_id}: {memory_used_mb:5.0f}MB/{total_mb:5.0f}MB ({memory_percent:5.1f}%) - {free_mb:5.0f}MB free {status}")
        
        selected_gpu_ids = [gpu_id for gpu_id, _, _, _, _, _ in selected_gpus]
        
        # Set CUDA_VISIBLE_DEVICES if not already set by external launcher
        if 'CUDA_VISIBLE_DEVICES' not in os.environ or os.environ.get('ACCELERATE_USE_CUDA', False):
            cuda_visible = ','.join(map(str, selected_gpu_ids))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible
            log.info(f"Set CUDA_VISIBLE_DEVICES={cuda_visible}")
        else:
            log.info(f"CUDA_VISIBLE_DEVICES already set: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
        return selected_gpu_ids
        
    except Exception as e:
        log.error(f"Error in auto GPU selection: {e}")
        return list(range(num_gpus))  # Fallback to default