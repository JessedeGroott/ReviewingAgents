# TEST_READY: Reasoning Probe & Representation Steering Test Suite

**Status**: E2E TEST SUITE READY  
**Author**: `test_writer_e2e`  
**Date**: 2026-10-10  
**Test Framework**: PyTorch 2.14.1 + pytest 9.1.1 + NumPy 2.4.4  
**Integrity Mode**: Opaque-box requirement-driven verification  

---

## 1. Test Runner Command

To run the complete E2E test suite across all 4 tiers:

```bash
pytest -v tests/e2e/
```

To run a specific test tier:

```bash
# Tier 1: Feature Coverage
pytest -v tests/e2e/test_tier1_features.py

# Tier 2: Boundary & Corner Cases
pytest -v tests/e2e/test_tier2_boundaries.py

# Tier 3: Pairwise Cross-Feature Interactions
pytest -v tests/e2e/test_tier3_pairwise.py

# Tier 4: Real-World Reasoning Scenarios
pytest -v tests/e2e/test_tier4_scenarios.py
```

---

## 2. Test Inventory & Summary Counts

| Tier | Test File | Description | Target | Actual Tests |
|---|---|---|:---:|:---:|
| **Tier 1** | `tests/e2e/test_tier1_features.py` | Feature Coverage (5 tests per feature across 7 features) | $\ge 35$ | **35** |
| **Tier 2** | `tests/e2e/test_tier2_boundaries.py` | Boundary, Corner & Resource Lifecycles (5 per feature) | $\ge 35$ | **35** |
| **Tier 3** | `tests/e2e/test_tier3_pairwise.py` | Cross-Feature Pairwise Integration Contracts | $\ge 7$ | **8** |
| **Tier 4** | `tests/e2e/test_tier4_scenarios.py` | Real-World Multi-Step Reasoning Scenarios (5 domains) | $\ge 5$ | **5** |
| **Total** | | **Complete Opaque-Box E2E Test Suite** | $\ge 82$ | **83** |

---

## 3. Feature Verification Checklist

| # | Feature | Spec Ref | Tier 1 Tests | Tier 2 Boundaries | Tier 3 Pairwise | Tier 4 Scenario |
|---|---------|:--------:|:------------:|:-----------------:|:---------------:|:---------------:|
| 1 | **Model Loader & Mock Transformer** | R3 | 5 | 5 | Pair 1, 8 | Scenario 1, 3 |
| 2 | **Internal Activation Hooking & Slicing** | R1 | 5 | 5 | Pair 1, 2, 5 | Scenario 1, 2, 3, 4, 5 |
| 3 | **Linear Probing & Calibration** | R1 | 5 | 5 | Pair 2, 3 | Scenario 1, 2, 4 |
| 4 | **Hallucination Drift Tracking** | R1 | 5 | 5 | Pair 5 | Scenario 2, 3, 5 |
| 5 | **Representation Steering Engine** | R2 | 5 | 5 | Pair 3, 4, 6 | Scenario 1, 2, 3, 4, 5 |
| 6 | **Dynamic Steering Controller & Generator** | R2, AC1 | 5 | 5 | Pair 4, 7, 8 | Scenario 1, 2, 3, 4, 5 |
| 7 | **Benchmark Dataset & Evaluation Suite** | R4, AC2 | 5 | 5 | Pair 6, 7 | Scenario 1, 4, 5 |

---

## 4. Multi-Step Reasoning Domains (Tier 4 Scenarios)

1. **`multi_hop_factual`**: Multi-hop relational chaining (Franz Kafka $\rightarrow$ birthplace Prague $\rightarrow$ Czech Republic $\rightarrow$ Czech language). Verifies ground-truth factual alignment and anti-hallucination steering.
2. **`temporal_chronology`**: Historical chronology & overlapping lifespans (Abraham Lincoln vs electric telephone patent). Verifies step-level drift detection at divergence step index ($s^* = 1$).
3. **`symbolic_arithmetic`**: Multi-step percentage discount and taxation word problem ($100 jacket with 20% discount and 8% tax $\rightarrow$ $86.40). Verifies numeric tolerance ($86.40 \pm 0.10$).
4. **`causal_scientific`**: Natural science state change derivation (continuous sub-zero cooling of water $\rightarrow$ open hexagonal crystal lattice $\rightarrow$ freezing and volume expansion).
5. **`common_misconceptions`**: TruthfulQA counterfactual lure rejection (Canberra vs Sydney lure). Verifies base model perplexity ratio ($\rho_{\text{PPL}} \le 1.35$) and n-gram distinctness ($\text{Dist}_2 \ge 0.50$).

---

## 5. Progressive Testability Architecture

- **In-Memory Mock Infrastructure**: `tests/conftest.py` provides an in-memory `MockTransformerModel` (4 layers, 64 hidden dim, 128 vocab) and `MockTokenizer` that execute the entire 83-test suite in **under 1.0 second on CPU** without downloading Hugging Face model weights.
- **Dynamic Module Discovery**: Tests utilize dynamic import resolvers (`get_probe_cls`, `get_steering_controller_cls`, etc.) supporting both submodule layouts (`reasoning_probe_steering.probes.linear_probe`) and top-level layouts.
- **Milestone-Aware Skip Markers**: Unimplemented modules from in-progress milestones (M1, M2, M3, M4) automatically skip with explicit descriptive reasons (e.g. `skip_if_no_probes`). As each milestone worker implements its modules, the corresponding tests un-skip and run automatically without requiring test modifications.

---

## 6. Baseline Verification Run

```
Platform: Android / Termux aarch64 (Python 3.14.6, Pytest 9.1.1, PyTorch 2.14.1, NumPy 2.4.4)
Command: pytest -v tests/e2e/
Result: 83 items collected, 9 passed, 74 skipped in 0.83s (0 failures, 0 errors)
```
