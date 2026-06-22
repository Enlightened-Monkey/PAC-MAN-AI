"""
device_helper.py – Helper for dynamic compute device selection (CPU/GPU).

Supports automatic detection of:
  1. NVIDIA CUDA / AMD ROCm (via torch.cuda)
  2. AMD GPU on native Windows (via torch-directml)
  3. CPU (fallback)
"""

import torch

def get_best_device() -> torch.device:
    """
    Return the best available compute device.
    """
    # 1. Check for NVIDIA CUDA or AMD ROCm
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # 2. Check for AMD GPU on Windows (DirectML)
    try:
        import torch_directml
        # DirectML may be available but requires fetching a device instance
        device = torch_directml.device()
        return device
    except ImportError:
        pass
        
    # 3. Fallback to CPU
    return torch.device("cpu")

def get_best_device_name() -> str:
    """
    Return the name string of the best available compute device.
    """
    device = get_best_device()
    if device.type == "privateuseone":  # characteristic type for DirectML
        return "directml"
    return device.type
