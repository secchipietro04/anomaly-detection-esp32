# true stft and feature extraction matching esp32 firmware (esp_fft_wrapper and inferencer.c)
from typing import Tuple, Optional, Union
import numpy as np

FFT_WINDOW_SIZE = 256
FFT_HOP_SIZE = 64
FFT_SIZE = 256
SAMPLES_PER_SEGMENT = 4096
DEFAULT_BINS = 128  # 128 accel + 128 gyro = 256 total bins

def compute_stft_spectrogram_channel(
    signal: Union[np.ndarray, list],
    window_size: int = FFT_WINDOW_SIZE,
    hop_size: int = FFT_HOP_SIZE,
    fft_size: int = FFT_SIZE,
    out_bins: int = DEFAULT_BINS
) -> np.ndarray:
    """
    Computes STFT magnitude spectrogram matching ESP32 esp_fft_wrapper.c:
    - Hann window matching dsps_wind_hann_f32
    - Complex FFT of size fft_size
    - Magnitude sqrt(re^2 + im^2) of first fft_size // 2 bins
    - 1D max pooling if out_bins < fft_size // 2 (matching pool_1d_max_pow2)
    - log1p compression: ln(1 + x) (matching dsp_fast_log1p_vec)
    Returns: (num_frames, out_bins)
    """
    sig = np.asarray(signal, dtype=np.float32).ravel()
    if len(sig) < window_size:
        sig = np.pad(sig, (0, window_size - len(sig)))

    num_frames = max(1, (len(sig) - window_size) // hop_size + 1)
    freq_bins = fft_size // 2  # 128 bins for 256 FFT

    # Hann window: w[i] = 0.5 * (1 - cos(2 * pi * i / (N - 1)))
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(window_size, dtype=np.float32) / (window_size - 1)))

    spec = np.zeros((num_frames, out_bins), dtype=np.float32)
    for f in range(num_frames):
        start = f * hop_size
        windowed = sig[start : start + window_size] * hann
        # rfft has freq_bins + 1 values (0..128), esp_fft_wrapper takes 0..127
        fft_complex = np.fft.rfft(windowed, n=fft_size)[:freq_bins]
        mag = np.abs(fft_complex).astype(np.float32)

        # Max pooling if out_bins < freq_bins
        if out_bins < freq_bins:
            K = freq_bins // out_bins
            mag = mag[:out_bins * K].reshape(out_bins, K).max(axis=1)
        elif out_bins > freq_bins:
            mag = np.pad(mag, (0, out_bins - freq_bins))

        # log1p compression matching dsp_fast_log1p_vec: ln(1 + max(0, x))
        spec[f] = np.log1p(np.maximum(mag, 0.0))

    return spec

def compute_segment_spectrogram(
    ax: Union[np.ndarray, list],
    ay: Union[np.ndarray, list],
    az: Union[np.ndarray, list],
    gx: Union[np.ndarray, list],
    gy: Union[np.ndarray, list],
    gz: Union[np.ndarray, list],
    accel_bins: int = DEFAULT_BINS,
    gyro_bins: int = DEFAULT_BINS,
    window_size: int = FFT_WINDOW_SIZE,
    hop_size: int = FFT_HOP_SIZE,
    fft_size: int = FFT_SIZE
) -> np.ndarray:
    """
    Matches inferencer.c compute_signal_features and run_ensemble_inference:
    1. Euclidean magnitude for accel and gyro:
       accel_mag = sqrt(ax^2 + ay^2 + az^2)
       gyro_mag = sqrt(gx^2 + gy^2 + gz^2)
    2. STFT on accel_mag and gyro_mag
    3. Concatenate along frequency axis: [accel_spec, gyro_spec]
    Returns: (num_frames, accel_bins + gyro_bins)
    For a 4096-sample segment with default parameters, returns (61, 256).
    """
    ax = np.asarray(ax, dtype=np.float32).ravel()
    ay = np.asarray(ay, dtype=np.float32).ravel()
    az = np.asarray(az, dtype=np.float32).ravel()
    gx = np.asarray(gx, dtype=np.float32).ravel()
    gy = np.asarray(gy, dtype=np.float32).ravel()
    gz = np.asarray(gz, dtype=np.float32).ravel()

    min_len = max(len(ax), len(ay), len(az), len(gx), len(gy), len(gz))
    if min_len == 0:
        return np.zeros((1, accel_bins + gyro_bins), dtype=np.float32)

    def _ensure_len(arr: np.ndarray, target: int) -> np.ndarray:
        if len(arr) < target:
            return np.pad(arr, (0, target - len(arr)))
        return arr[:target]

    ax = _ensure_len(ax, min_len)
    ay = _ensure_len(ay, min_len)
    az = _ensure_len(az, min_len)
    gx = _ensure_len(gx, min_len)
    gy = _ensure_len(gy, min_len)
    gz = _ensure_len(gz, min_len)

    accel_mag = np.sqrt(ax * ax + ay * ay + az * az)
    gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    accel_spec = compute_stft_spectrogram_channel(
        accel_mag, window_size=window_size, hop_size=hop_size, fft_size=fft_size, out_bins=accel_bins
    )
    gyro_spec = compute_stft_spectrogram_channel(
        gyro_mag, window_size=window_size, hop_size=hop_size, fft_size=fft_size, out_bins=gyro_bins
    )

    return np.concatenate([accel_spec, gyro_spec], axis=-1)
