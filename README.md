# Reasoning Probe & Representation Steering Prototype

A research prototype for open-weights transformer models that hooks into intermediate representations (residual streams and attention head outputs) during autoregressive reasoning, trains linear probes to detect latent truthfulness and hallucination drift, and applies inference-time representation steering (Contrastive Activation Addition / Inference-Time Intervention) to mitigate reasoning hallucinations.

---

## 1. Overview & Architecture

Modern large reasoning models can suffer from **hallucination in internal reasoning traces**, where extended deliberation accumulates subtle errors, exhibits flaw repetition, or rationalizes false conclusions. This prototype provides an end-to-end framework to:
1. **Harvest intermediate activations** at token-level granularity during forward passes.
2. **Train calibrated linear probes** (Logistic, Ridge, Mass-Mean centroid difference) to score latent veracity and monitor real-time reasoning drift.
3. **Extract contrastive steering vectors** via Contrastive Activation Addition (CAA) and Fisher covariance whitening.
4. **Intervene dynamically at inference time** via non-destructive PyTorch forward hooks with norm-preserving and variance scaling controls.
5. **Evaluate generation quality comparatively** across factual reasoning benchmarks, logging Hallucination Reduction Rate (HRR), probe ROC-AUC, and base-model perplexity.

### Codebase Layout

```
reasoning_probe_steering/
├── reasoning_probe_steering/       # Core library package
│   ├── models/                    # Model loader & deterministic MockTransformer
│   │   ├── loader.py              # HF & Mock model loading dispatch
│   │   └── mock_transformer.py    # Architectural mock transformer for fast testing
│   ├── hooks/                     # Low-level PyTorch forward hook handlers
│   │   ├── collector.py           # ActivationCollector for residual & head activations
│   │   └── steering_hook.py       # ResidualSteeringHook & AttentionHeadSteeringHook
│   ├── probes/                    # Linear probing and drift tracking
│   │   ├── linear_probe.py        # ActivationProbe, RidgeProbe, MassMeanProbe
│   │   ├── calibration.py         # TemperatureScaling & Expected Calibration Error (ECE)
│   │   └── drift_tracker.py       # Real-time token trajectory drift monitor
│   ├── steering/                  # Representation steering engine
│   │   ├── vector_extractor.py    # SteeringVectorExtractor (CAA, probe normal, Fisher)
│   │   ├── controller.py          # SteeringController (multi-layer intervention & cleanup)
│   │   └── generator.py           # SteeredGenerator (unsteered vs. steered comparison)
│   └── benchmark/                 # Evaluation dataset & metrics suite
│       ├── dataset.py             # FactualReasoningDataset across 5 core domains
│       ├── verifier.py            # Deterministic ground truth validators
│       ├── metrics.py             # ROC-AUC, PR-AUC, HRR, Perplexity, Distinct n-grams
│       └── evaluator.py           # ComparativeEvaluator harness
├── tests/                         # Comprehensive 4-tier test suite
│   ├── unit/                      # Modular unit tests (58+ tests)
│   ├── e2e/                       # 4 tiers of end-to-end integration tests (Tier 1-4)
│   ├── stress/                    # Dimension & boundary stress test suites
│   ├── test_adversarial_m1..m4.py # Adversarial matrix rank & collinearity stress tests
│   └── conftest.py                # Deterministic test fixtures & mock models
├── cli.py                         # Unified command-line interface
├── pyproject.toml                 # Packaging & dependencies
├── pytest.ini                     # Test configuration
└── README.md                      # Complete reproduction & usage documentation
```

---

## 2. Mathematical Formulations

### 2.1 Linear Probing & Calibration
Given hidden activations $h_t^{(l)} \in \mathbb{R}^d$ at layer $l$:
- **Mass-Mean Centroid Difference:**
  $$\theta = \mu_{\text{truthful}} - \mu_{\text{hallucinated}}$$
- **Logistic Probing:**
  $$P(\text{truthful} \mid h) = \sigma(w^T h + b)$$
  Trained with Adam optimizer and binary cross-entropy loss.
- **Temperature Calibration:**
  Logits $z$ are scaled by learned temperature $T > 0$: $\hat{p} = \sigma(z / T)$ to minimize Expected Calibration Error (ECE):
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### 2.2 Representation Steering (CAA / ITI)
- **Fisher-Whitened Steering Vector:**
  $$\theta = (\Sigma_{\text{pooled}} + \lambda I)^{-1} (\mu_{\text{pos}} - \mu_{\text{neg}})$$
- **Forward-Pass Intervention:**
  At target layers $l \in L$, activations $h$ are displaced during generation:
  $$\tilde{h} = h + \alpha \cdot \frac{\theta}{\|\theta\|_2}$$
  Under **norm-preserving mode**, activations are rescaled to prevent runaway token embeddings:
  $$\tilde{h}_{\text{normed}} = \tilde{h} \cdot \frac{\|h\|_2}{\|\tilde{h}\|_2}$$

### 2.3 Hallucination Reduction Rate (HRR)
$$\text{HRR} = \frac{\text{Error Rate}_{\text{baseline}} - \text{Error Rate}_{\text{steered}}}{\text{Error Rate}_{\text{baseline}}}$$

---

## 3. Installation & Requirements

The codebase is engineered with a pure PyTorch and NumPy backend (zero compilation bottlenecks), supporting CPU and GPU execution out of the box:

```bash
# Clone and enter directory
cd ~/teamwork_projects/reasoning_probe_steering

# Install dependencies (or run in active Python environment with torch and pytest)
pip install torch numpy pytest
```

---

## 4. Quickstart CLI Usage

The CLI (`cli.py`) provides unified access to probe training, inference generation, and benchmark evaluation.

### 4.1 Train Activation Probes
Extract activations on factual dataset samples and train calibrated probes:

```bash
python cli.py train-probe \
  --model mock \
  --layer 1 \
  --probe-type logistic \
  --output-dir checkpoints/probes
```

*For real open-weights models:*
```bash
python cli.py train-probe \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layer 12 \
  --probe-type logistic \
  --device cpu
```

### 4.2 Generate Reasoning Traces (Baseline vs. Steered)
Generate side-by-side comparison with dynamic steering toggled on/off:

```bash
python cli.py generate \
  --model mock \
  --prompt "Which planet is known as the Red Planet?" \
  --compare \
  --alpha 1.5 \
  --max-new-tokens 16
```

To run without steering:
```bash
python cli.py generate \
  --model mock \
  --prompt "Explain Fermat's Little Theorem" \
  --no-steer
```

### 4.3 Run Comparative Benchmark Evaluation
Run the automated evaluation harness comparing baseline vs. steered generation across factual reasoning queries:

```bash
python cli.py eval \
  --model mock \
  --alpha 1.5 \
  --output-json benchmark_report.json \
  --output-csv benchmark_report.csv
```

Outputs include:
- `benchmark_report.json`: Detailed sample-level generation traces and aggregate metrics.
- `benchmark_report.csv`: Flat evaluation metrics summary.
- Terminal formatted scorecard showing Accuracy Gain, Error Delta, HRR, Mean Perplexity, and 2-Gram diversity.

---

## 5. Programmatic Python API

### 5.1 Activation Harvesting & Hooking
```python
from reasoning_probe_steering.models import load_model_and_tokenizer
from reasoning_probe_steering.hooks import ActivationCollector

model, tokenizer = load_model_and_tokenizer("mock", device="cpu")

# Context manager collects activations without memory leaks
with ActivationCollector(model, target_layers=[1], token_position="last") as collector:
    inputs = tokenizer("What is the speed of light?", return_tensors="pt")
    outputs = model(**inputs)
    activations = collector.get_activations(layer=1)  # Tensor of shape (batch, hidden_dim)
```

### 5.2 Dynamic Activation Steering
```python
import torch
from reasoning_probe_steering.steering import SteeringController, SteeredGenerator

# Define or load extracted steering vector
steering_vector = torch.randn(64)

controller = SteeringController(model, target_layers=[1], device="cpu")
controller.register_vector(layer=1, vector=steering_vector)

generator = SteeredGenerator(model, tokenizer, controller=controller)

# Compare baseline vs steered output
comparison = generator.compare(
    prompt="Who discovered penicillin?",
    alpha=1.5,
    max_new_tokens=16
)
print("Baseline:", comparison.baseline.text)
print("Steered: ", comparison.steered.text)
```

---

## 6. Testing & Verification

The test infrastructure includes over 540 automated tests across 4 tiers:
- **Unit Tests:** `tests/unit/` (models, hooks, probes, calibration, drift tracking, steering, generator, benchmark, CLI).
- **E2E Integration Tests:** `tests/e2e/` (Features 1 through 7, boundary edge cases, pairwise component interactions, multi-step scenario tests).
- **Stress & Adversarial Tests:** `tests/stress/` and `tests/test_adversarial_*.py` (rank-deficient matrices, $D=4096$ activations, 98:2 class imbalance).

### Running Tests
Execute the entire test suite:

```bash
pytest -v tests/
```

**Expected Result:**
```
============================= 542 passed in ~32s =============================
```
Zero failures, zero skipped tests.

---

## 7. License & Citation

Developed as an autonomous multi-agent research prototype for the investigation and mitigation of internal reasoning hallucinations in large language models.
