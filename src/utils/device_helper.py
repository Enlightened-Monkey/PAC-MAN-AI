"""
device_helper.py – Pomocnik do dynamicznego wyboru urządzenia obliczeniowego (CPU/GPU).

Wspiera automatyczne wykrywanie:
  1. NVIDIA CUDA / AMD ROCm (poprzez torch.cuda)
  2. AMD GPU na natywnym systemie Windows (poprzez torch-directml)
  3. CPU (jako fall-back)
"""

import torch

def get_best_device() -> torch.device:
    """
    Zwraca najlepsze dostępne urządzenie do obliczeń.
    """
    # 1. Sprawdzenie NVIDIA CUDA lub AMD ROCm
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # 2. Sprawdzenie AMD GPU na Windows (DirectML)
    try:
        import torch_directml
        # DirectML może być dostępne, ale wymaga pobrania instancji urządzenia
        device = torch_directml.device()
        return device
    except ImportError:
        pass
        
    # 3. Fall-back do CPU
    return torch.device("cpu")

def get_best_device_name() -> str:
    """
    Zwraca tekstową nazwę najlepszego dostępnego urządzenia.
    """
    device = get_best_device()
    if device.type == "privateuseone": # Charakterystyczny typ dla DirectML
        return "directml"
    return device.type
