# freegrad

Experimental JAX training runtime for composable learning-stack and model studies.

Core rule:

- component folders define reusable pieces
- execution drivers own physical evaluation, micro-batching, and train chunks
- learning stacks compose probe, conditioning, step-size, repair/reset, and update policies
- runtime binds pieces together and schedules runs
- specs define concrete experimental choices

Initial bootstrap includes:

- MNIST data loading and preprocessing
- generic SGD/Adam-style learning stacks
- pure JAX SimpleMLP and ModernSmallCNN models
- local JSONL tracking, pickle checkpointing, and study expansion

Note: This repo is only set up for unconstrained problems.

## Quick start

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run tests:

```bash
pytest
```

Run a condition:

```bash
python -m scripts.run_condition suites.mnist.conditions.mlp_adam:make_condition
```

Run a study:

```bash
python -m scripts.run_study suites.mnist.adam_comparison_study:make_study
```

Watch run status in a separate terminal (Windows, macOS, or Linux):

```bash
python -m scripts.watch_study results/mnist_adam_model_comparison
```

The dashboard refreshes every two seconds and stops with `Ctrl+C`. Use
`--once` to print one snapshot without watching.

Study runs write a `study.json` manifest at the study root. Rerunning the same study name/output path skips runs already marked `COMPLETED` and resumes incomplete or failed runs from their latest checkpoint when one exists.

## Training config

Training runs distinguish four batch meanings:

- model example shape excludes the leading batch axis
- `mini_batch_size`: execution micro-batch size chosen for memory/device limits
- `macro_batch_size`: effective batch size consumed by one learning-stack update
- `eval_macro_batch_size`: validation/reporting batch size used by evaluation runners

The main knobs are:

- `mini_batch_size`: training execution micro-batch size.
- `macro_batch_size`: examples consumed per effective-batch update. This must be divisible by `mini_batch_size`.
- `train_chunk_size`: number of train steps grouped into one compiled train-chunk execution.
- `eval_mini_batch_size`: validation execution micro-batch size.
- `eval_macro_batch_size`: examples consumed per validation step. This must be divisible by `eval_mini_batch_size`.

The runtime hands these sizes to execution drivers when the run starts, then schedules completed train chunks and validation passes. Train metrics and run status are emitted once per completed train chunk. Validation runs only between chunks, with `eval_every` measured in completed train chunks. `n_chunks_per_checkpoint` saves checkpoints every N completed train chunks.
