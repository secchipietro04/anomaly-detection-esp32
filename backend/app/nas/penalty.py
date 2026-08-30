# adaptive ram penalty calculation based on 75% free ram budget
from typing import Optional, List
from app.config import get_settings

DEFAULT_FALLBACK_RAM_BUDGET = 4 * 1024 * 1024  # 4MB fallback if no health reported yet

def calculate_ram_budget(ram_free: Optional[int] = None) -> int:
    # 75% of sensor's reported free ram
    if ram_free is not None and ram_free > 0:
        return int(ram_free * 0.75)
    settings = get_settings()
    return settings.ram_limit_bytes or DEFAULT_FALLBACK_RAM_BUDGET

def calculate_memory_penalty(
    total_size_bytes: int,
    ram_free: Optional[int] = None,
    lambda_factor: Optional[float] = None,
    per_mb: bool = True
) -> float:
    # zero penalty if within ram budget, proportional latency penalty if exceeding
    settings = get_settings()
    budget = calculate_ram_budget(ram_free)
    factor = lambda_factor if lambda_factor is not None else settings.sd_latency_penalty_factor

    if total_size_bytes <= budget:
        return 0.0

    excess_bytes = total_size_bytes - budget
    if per_mb:
        excess_mbs = excess_bytes / (1024.0 * 1024.0)
        return float(factor * excess_mbs)
    return float(factor * excess_bytes)

def calculate_ensemble_size(
    router_size: int,
    memory_size: int = 0,
    autoencoder_sizes: Optional[List[int]] = None
) -> int:
    # clean single-line sum of all ensemble submodel sizes
    ae_sum = sum(autoencoder_sizes) if autoencoder_sizes else 0
    return router_size + memory_size + ae_sum
