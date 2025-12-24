<div align="center">

# Beyond High-Entropy Exploration: Correctness-Aware Low-Entropy Segment-Based Advantage Shaping for Reasoning LLMs.


</div>
🔍 Project Overview
This project proposes LESS (Low-Entropy Segment Shaping), a correctness-aware reinforcement learning framework for reasoning LLMs. Unlike existing entropy-based methods that only focus on high-entropy exploration tokens, LESS explicitly leverages low-entropy segments—stable structural components that account for ~80% of reasoning trajectories—to optimize policy updates.
By distinguishing low-entropy segments into "correct-only", "incorrect-only", and "shared" types, LESS amplifies productive reasoning patterns, suppresses repeated errors, and preserves high-entropy exploration. Instantiated on top of GRPO, it consistently improves accuracy, stability, and robustness across mathematical reasoning tasks.
🚀 Core Features
Low-Entropy Segment Awareness: Identifies and modulates stable reasoning structures based on their correlation with correctness.
Plug-and-Play Design: Seamlessly integrates with existing RLVR algorithms (e.g., GRPO) as an advantage-shaping module.
Performance Boost: Outperforms strong baselines (GRPO, Forking Tokens, KL-Cov) on 6+ math benchmarks.
Robustness Enhancement: Reduces worst-case performance variance and raises the floor of model reliability.
Broad Compatibility: Works with diverse model scales (1.5B–7B) and both math-specialized/base LLMs.
📊 Key Results
Accuracy Improvement: Average accuracy gains of 2–4 points across Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, and Qwen2.5-Base-7B.
Challenging Tasks: Notable gains on AIME24/25 (math olympiad-level tasks) and AMC23.
Worst-Case Robustness: Improves worst@32 score by +6.1 (1.5B) and +7.8 (7B) points compared to vanilla GRPO.
Stability: Reduces response variance (std@32) across sampled rollouts for more predictable reasoning.
🛠️ Getting Started
Prerequisites
Dependencies: Follow verl installation guide (supports PyTorch, FSDP, Megatron-LM).
Inference Engine: vLLM ≥ 0.8.2 (for high-throughput rollout generation).
Model: Qwen2.5 family (1.5B/7B Math, 7B Base; other LLMs compatible with modification).
Dataset: hendrycks_math (7.5k math problems, covers algebra, geometry, number theory, etc.).
Training Configuration
Recommended hyperparameters:
Minimum low-entropy segment length μ=5 (optimal for stability and accuracy).
Entropy quantile h (follows Wang et al., 2025 for threshold calculation).
Batch size: 512, Learning rate: 1e-6, Clip range: 0.2–0.28.
Quick Run
For training Qwen2.5-7B on a single node (8 NVIDIA A100-40G GPUs):
bash
运行
cd verl
conda activate your_env
# Configure dataset path and hyperparameters in 7b_base.sh first
bash 7b_base.sh
📋 Acknowledgement
We build the RL framework on top of verl (HybridFlow), a flexible and efficient RLHF library.
Inference is accelerated by vLLM with PagedAttention for high throughput.
Models are trained on the Qwen2.5 family (Alibaba Cloud), optimized for mathematical reasoning.
Training data is derived from hendrycks_math, a benchmark for measuring mathematical problem-solving ability.
📝 Citation
If you find this work useful, please cite our paper:
bibtex
@article{chen2025less,
  title={Beyond High-Entropy Exploration: Correctness-Aware Low-Entropy Segment-Based Advantage Shaping for Reasoning LLMs},
  author={Chen, Xinzhu and Li, Xuesheng and Sun, Zhongxiang and Yu, Weijie},
  journal={arXiv preprint arXiv:2512.00908v1},
  year={2025}
}
