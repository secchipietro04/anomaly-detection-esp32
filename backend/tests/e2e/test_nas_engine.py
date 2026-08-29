# tier 3: nas engine - op filter and memory penalty logic
import pytest
from app.nas.op_filter import validate_candidate_architecture, ALL_SUPPORTED_OPS
from app.nas.penalty import calculate_memory_penalty, calculate_ensemble_size


# --- op filter ---

def test_valid_ops_pass():
    required = ["FULLY_CONNECTED", "RELU", "SOFTMAX"]
    enabled = list(ALL_SUPPORTED_OPS)
    valid, missing = validate_candidate_architecture(required, enabled)
    assert valid is True
    assert missing == []


def test_missing_op_pruned():
    required = ["FULLY_CONNECTED", "UNSUPPORTED_WEIRD_OP"]
    enabled = ["FULLY_CONNECTED", "RELU"]
    valid, missing = validate_candidate_architecture(required, enabled)
    assert valid is False
    assert "UNSUPPORTED_WEIRD_OP" in missing


def test_empty_required_always_valid():
    valid, missing = validate_candidate_architecture([], list(ALL_SUPPORTED_OPS))
    assert valid is True


# --- memory penalty ---

def test_no_penalty_under_4mb():
    ram_limit = 4 * 1024 * 1024  # 4MB
    total = 3 * 1024 * 1024  # 3MB
    penalty = calculate_memory_penalty(total, ram_limit, lambda_factor=0.05, per_mb=True)
    assert penalty == 0.0


def test_penalty_above_4mb():
    ram_limit = 4 * 1024 * 1024
    total = 5 * 1024 * 1024  # 5MB -> 1MB over limit
    penalty = calculate_memory_penalty(total, ram_limit, lambda_factor=0.05, per_mb=True)
    assert penalty > 0.0


def test_ensemble_size_sum():
    size = calculate_ensemble_size(router_size=100, memory_size=200, autoencoder_sizes=[300, 400])
    assert size == 1000


def test_no_memory_model_size():
    size = calculate_ensemble_size(router_size=500, memory_size=0, autoencoder_sizes=[256])
    assert size == 756
