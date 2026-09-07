# ESP32 hardware latency estimator, 4MB RAM cache simulation, and SD model swap model
from typing import Optional, List, Dict, Any, Tuple

ESP32_CLOCK_HZ = 240_000_000.0  # 240 MHz Xtensa dual-core
ESP32_OPTIM_FACTOR = 0.7        # SIMD / TFLite Micro Xtensa DSP speedup factor
RAM_CACHE_BUDGET_BYTES = 4 * 1024 * 1024  # 4MB RAM cache budget
SD_LOAD_SPEED_BPS = 1_000_000.0  # 1.0 MB/s effective SD card read throughput
MIN_APPRECIABLE_FREQ_HZ = 0.5    # 0.5 Hz minimum execution frequency (max 2.0s per segment)

def estimate_model_macs(param_count: int, num_eval_frames: int = 6) -> float:
    # estimates MAC operations per segment evaluation
    return float(param_count * num_eval_frames)

def calculate_esp32_execution_profile(
    router_size: int,
    router_macs: float,
    ae_sizes: List[int],
    ae_macs: List[float],
    memory_size: int = 0,
    memory_macs: float = 0.0,
    ram_limit_bytes: int = RAM_CACHE_BUDGET_BYTES,
    sd_speed_bps: float = SD_LOAD_SPEED_BPS
) -> Dict[str, Any]:
    # 1. total ensemble size
    total_size_bytes = router_size + memory_size + sum(ae_sizes)
    num_submodels = len(ae_sizes)

    # 2. base inference computation time on ESP32 (MACs / clock * 0.7)
    avg_ae_macs = (sum(ae_macs) / max(1, num_submodels)) if ae_macs else 0.0
    total_segment_macs = router_macs + memory_macs + avg_ae_macs
    t_inference_sec = (total_segment_macs / ESP32_CLOCK_HZ) * ESP32_OPTIM_FACTOR

    # 3. 4MB RAM cache and SD deload/reload time model
    # If total models <= 4MB, everything is resident in RAM cache -> zero SD I/O
    if total_size_bytes <= ram_limit_bytes:
        cache_miss_prob = 0.0
        t_sd_reload_sec = 0.0
    else:
        # Cache miss probability: fraction of models that cannot fit in 4MB RAM simultaneously
        cache_miss_prob = max(0.0, 1.0 - (float(ram_limit_bytes) / float(total_size_bytes)))
        avg_ae_size = (sum(ae_sizes) / max(1, num_submodels)) if ae_sizes else 0.0
        # Time to deload and reload a model from SD at 1MB/s
        t_model_swap = avg_ae_size / sd_speed_bps
        t_sd_reload_sec = cache_miss_prob * t_model_swap

    # 4. total execution cycle time and frequency
    total_time_sec = t_inference_sec + t_sd_reload_sec
    exec_frequency_hz = 1.0 / max(1e-5, total_time_sec)

    # 5. penalty for failing 0.5 Hz minimum requirement
    freq_penalty = 0.0
    if exec_frequency_hz < MIN_APPRECIABLE_FREQ_HZ:
        freq_deficit = MIN_APPRECIABLE_FREQ_HZ - exec_frequency_hz
        freq_penalty = 50.0 * (freq_deficit ** 2) + 10.0 * max(0.0, total_time_sec - (1.0 / MIN_APPRECIABLE_FREQ_HZ))

    return {
        "total_size_bytes": total_size_bytes,
        "fits_in_ram": total_size_bytes <= ram_limit_bytes,
        "cache_miss_prob": cache_miss_prob,
        "t_inference_sec": t_inference_sec,
        "t_sd_reload_sec": t_sd_reload_sec,
        "total_time_sec": total_time_sec,
        "exec_frequency_hz": exec_frequency_hz,
        "freq_penalty": freq_penalty
    }

def calculate_ensemble_size(
    router_size: int,
    memory_size: int = 0,
    autoencoder_sizes: Optional[List[int]] = None
) -> int:
    ae_sum = sum(autoencoder_sizes) if autoencoder_sizes else 0
    return router_size + memory_size + ae_sum

def calculate_ram_budget(ram_free: int, factor: float = 0.75) -> int:
    return int(ram_free * factor)

def calculate_memory_penalty(total_size: int, ram_free: int, lambda_factor: float = 0.05) -> float:
    budget = calculate_ram_budget(ram_free)
    if total_size <= budget:
        return 0.0
    excess_mb = (total_size - budget) / (1024 * 1024)
    return float(excess_mb * lambda_factor)

