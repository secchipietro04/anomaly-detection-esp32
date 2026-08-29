# memory penalty calculation for nas models
from typing import List, Optional, Union, Dict, Any
from app.config import get_settings

RAM_LIMIT_BYTES_DEFAULT = 4 * 1024 * 1024  # 4MB limit
SD_PENALTY_FACTOR_DEFAULT = 0.05

def calculate_ensemble_size(
    router_size: int,
    memory_size: int = 0,
    autoencoder_sizes: Optional[List[int]] = None
) -> int:
    # calculate total size in bytes for the ensemble
    total = int(router_size) + int(memory_size)
    if autoencoder_sizes:
        total += sum(int(s) for s in autoencoder_sizes)
    return total

def calculate_memory_penalty(
    total_size_bytes: int,
    ram_limit_bytes: Optional[int] = None,
    lambda_factor: Optional[float] = None,
    per_mb: bool = True
) -> float:
    # calculate penalty if ensemble size exceeds 4MB
    settings = get_settings()
    limit = ram_limit_bytes if ram_limit_bytes is not None else settings.ram_limit_bytes
    factor = lambda_factor if lambda_factor is not None else settings.sd_latency_penalty_factor
    
    if total_size_bytes <= limit:
        # no penalty when running in sram
        return 0.0
    
    excess_bytes = total_size_bytes - limit
    if per_mb:
        # scale excess by megabytes
        excess_mb = excess_bytes / (1024.0 * 1024.0)
        return float(factor * excess_mb)
    else:
        # raw byte difference penalty
        return float(factor * excess_bytes)

def evaluate_models_memory_penalty(
    models: List[Union[int, bytes, Dict[str, Any], Any]],
    ram_limit_bytes: Optional[int] = None,
    lambda_factor: Optional[float] = None,
    per_mb: bool = True
) -> float:
    # compute total bytes from model objects or sizes and get penalty
    total_bytes = 0
    for m in models:
        if isinstance(m, int):
            total_bytes += m
        elif isinstance(m, (bytes, bytearray)):
            total_bytes += len(m)
        elif isinstance(m, dict):
            if "size_bytes" in m:
                total_bytes += int(m["size_bytes"])
            elif "data" in m and isinstance(m["data"], (bytes, bytearray)):
                total_bytes += len(m["data"])
            elif "tflite_binary" in m and isinstance(m["tflite_binary"], (bytes, bytearray)):
                total_bytes += len(m["tflite_binary"])
        elif hasattr(m, "data") and isinstance(m.data, (bytes, bytearray)):
            total_bytes += len(m.data)
        elif hasattr(m, "tflite_binary") and isinstance(m.tflite_binary, (bytes, bytearray)):
            total_bytes += len(m.tflite_binary)
        elif hasattr(m, "size_bytes") and isinstance(m.size_bytes, int):
            total_bytes += m.size_bytes
            
    return calculate_memory_penalty(
        total_size_bytes=total_bytes,
        ram_limit_bytes=ram_limit_bytes,
        lambda_factor=lambda_factor,
        per_mb=per_mb
    )
