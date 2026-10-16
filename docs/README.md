# Documentation

[项目研究与评测说明](project-evaluation.md) provides a detailed Chinese review of
the use case, training implementation, reported results, and reproducibility limits.

[Qwen3.5-9B 基线评测复现指南](reproduce-baseline-qwen3.5-9b.md) covers the first
reproduction step, including isolated dependencies, an Ascend serving reference,
development checks, and the Final-200 Clean baseline run.

Follow the guides in workflow order:

1. [Data collection](data-collection.md) explains how the checked-in SFT data
   was produced and audited.
2. [SFT](sft.md) trains the first useful shopping agent.
3. [GRPO](grpo.md) improves that model with online environment reward.
4. [Evaluation](evaluation.md) compares baseline, SFT and GRPO fairly.
5. [Final-200 Clean evaluation dataset](evaluation-dataset.md) defines the current
   curated benchmark and its update record.

[Reward v3](reward-v3.md) is the detailed specification shared by collection,
GRPO and evaluation.
