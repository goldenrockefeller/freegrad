# freegrad

Experimental JAX training runtime for optimizer and model studies.

Core rule:

- component folders define reusable pieces
- runtime binds pieces together and executes them
- specs define concrete experimental choices

Initial bootstrap includes:

- MNIST data loading and preprocessing
- manual Adam optimizer
- pure JAX SimpleMLP and ModernSmallCNN models
- local JSONL tracking, pickle checkpointing, and study expansion

Note: This repo is only set up for unconstrained problems.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Run a condition:

```bash
python scripts/run_condition.py specs.mnist.mlp_adam:make_condition
```

Run a study:

```bash
python scripts/run_study.py specs.mnist.adam_comparison_study:make_study
```

Study runs write a `study.json` manifest at the study root. Rerunning the same study name/output path skips runs already marked `COMPLETED` and resumes incomplete or failed runs from their latest checkpoint when one exists.

## Training config

Training runs now distinguish between the batching owned by the train step and the chunk cadence owned by the runtime:

- `mini_batch_size`: train-step mini-batch size used inside the batched train loss and metrics wrappers.
- `macro_batch_size`: examples consumed per train step. This must be divisible by `mini_batch_size`.
- `train_chunk_size`: number of train steps grouped into one compiled train-chunk execution.
- `eval_mini_batch_size`: validation mini-batch size used inside the batched eval metrics wrapper.
- `eval_macro_batch_size`: examples consumed per validation step. This must be divisible by `eval_mini_batch_size`.

The runtime hands these sizes to the train and validation owners when the run starts, then only schedules completed train chunks and validation passes. Train metrics and run status are emitted once per completed train chunk. Validation runs only between chunks, with `eval_every` measured in completed train chunks. `n_chunks_per_checkpoint` saves checkpoints every N completed train chunks.
