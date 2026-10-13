# Original User Request

## 2026-10-10T17:58:18Z

Build a research prototype that probes internal representations (hidden states and attention activations) and applies representation steering (such as Inference-Time Intervention or activation patching) to detect and mitigate hallucinations during multi-step reasoning in open-weights transformer models.

Working directory: ~/teamwork_projects/reasoning_probe_steering
Integrity mode: benchmark

## Requirements

### R1. Internal Activation Extraction & Probing
The system must hook into intermediate layers (hidden states and attention head outputs) during autoregressive generation to extract token-level representations across reasoning steps. It must support training and evaluating linear probes to score latent truthfulness and detect hallucination drift during reasoning.

### R2. Inference-Time Representation Steering
The system must provide an intervention mechanism that modifies or shifts activations along identified truthful / grounded directional vectors during the forward pass at inference time, enabling direct comparison of generated reasoning and answers with and without steering.

### R3. Open-Weights Model Support
The prototype must support standard lightweight open-weights models (such as Qwen2.5-0.5B or Llama-3.2-1B) with both CPU and GPU execution capabilities, without requiring distributed training infrastructure.

### R4. Automated Verification & Comparative Evaluation Suite
The system must provide a self-contained evaluation harness with a benchmark dataset of factual reasoning queries. The harness must measure and log:
1. Probe detection accuracy for known factual errors.
2. Comparative generation quality (hallucination reduction rate, factual accuracy, and linguistic coherence/perplexity) between baseline generation and steered generation.

## Acceptance Criteria

### Functionality & Testing
- [ ] Automated test suite (`pytest`) runs and passes completely, covering activation hooking, probe training/inference, and steering intervention logic.
- [ ] Activation steering can be toggled on/off dynamically via a clean programmatic API and CLI.

### Evaluation & Reproducibility
- [ ] End-to-end benchmark script runs on the target model and produces a structured report (JSON/CSV and console summary) comparing baseline vs. steered outputs.
- [ ] Steered generation demonstrates a measurable reduction in hallucinated reasoning steps on the evaluation set while maintaining readable and coherent reasoning text.
- [ ] Complete setup and reproduction instructions provided in a `README.md`.


## 2026-10-10T21:26:04Z

Continue and finish the existing research prototype in `~/teamwork_projects/reasoning_probe_steering`, which probes internal representations (hidden states and attention activations) and applies representation steering to detect and mitigate hallucinations in multi-step reasoning. Milestones 1-3 (model loading, hooks, probes, drift tracking, steering engine) are already implemented and audited; the previous team stopped partway through Milestone 4 because of a quota error, so assess the current state of the code first and complete what remains.

Working directory: ~/teamwork_projects/reasoning_probe_steering
Integrity mode: benchmark

## Requirements

### R1. Finish the evaluation harness and CLI
Complete the benchmark dataset, verifier, metrics, evaluator and CLI so that baseline and steered generation can be compared end to end, and steering can be toggled on and off from both the programmatic API and the CLI.

### R2. Full test suite passes
Every unit, end-to-end and adversarial test passes with no skipped tests left over from unfinished milestones.

### R3. Benchmark run and report
Run the end-to-end benchmark on the target open-weights model (Qwen2.5-0.5B-Instruct on CPU, or the closest available model) and produce a structured JSON/CSV report plus a console summary comparing baseline and steered outputs: probe detection accuracy, hallucination reduction rate, factual accuracy, and perplexity/coherence.

### R4. Documentation
Provide a `README.md` with setup and reproduction instructions, including how to run the tests, the benchmark and the CLI.

## Acceptance Criteria

### Testing
- [ ] `pytest` over the whole `tests/` directory exits 0 with zero skipped and zero failed tests.
- [ ] Steering can be enabled and disabled via the Python API and via CLI flags, covered by tests.

### Evaluation and Reproducibility
- [ ] The benchmark script runs end to end and writes a JSON or CSV report containing baseline vs. steered metrics.
- [ ] The report shows a measurable hallucination reduction rate for steered generation versus baseline, with perplexity reported alongside. If the target model cannot be loaded, the report states this explicitly and says which model was used instead.
- [ ] Following only the README, a fresh checkout can reproduce the test run and the benchmark.

## 2026-10-10T21:36:20Z

The quota has reset. Please start the task now: finish the reasoning_probe_steering project in ~/teamwork_projects/reasoning_probe_steering as described in your original prompt. Assess the current state first. The previous team was stopped at Milestone 4 and no other team is running now.
