# unit test for adaptive 75% free ram budget penalty
from app.nas.penalty import calculate_ram_budget, calculate_memory_penalty, calculate_ensemble_size

def test_ram_budget_75_percent():
    budget = calculate_ram_budget(ram_free=8 * 1024 * 1024)  # 8MB free
    assert budget == 6 * 1024 * 1024  # 6MB budget

def test_zero_penalty_within_budget():
    ram_free = 8 * 1024 * 1024  # 8MB free -> 6MB budget
    total_size = 5 * 1024 * 1024  # 5MB ensemble <= 6MB
    penalty = calculate_memory_penalty(total_size, ram_free=ram_free)
    assert penalty == 0.0

def test_penalty_when_exceeding_budget():
    ram_free = 8 * 1024 * 1024  # 6MB budget
    total_size = 7 * 1024 * 1024  # 7MB ensemble -> 1MB excess
    penalty = calculate_memory_penalty(total_size, ram_free=ram_free, lambda_factor=0.05)
    assert penalty == pytest.approx(0.05)

def test_clean_ensemble_size_sum():
    size = calculate_ensemble_size(router_size=1000, memory_size=2000, autoencoder_sizes=[3000, 4000])
    assert size == 10000
import pytest
