---
title: "Multi-LoRA Tinker Gateway"
description: "Serve concurrent LoRA fine-tuning clients on one shared base model through the Tinker protocol."
# Generated from examples/multi_lora/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
> **Read the docs:** [Multi-LoRA training](https://miles.radixark.com/docs/advanced/lora#multi-lora-training).

- `serve_qwen3_30b_a3b_tinker.py`: prepare Qwen3-30B-A3B and launch the gateway.
- `run_multi_tenant_example.py`: check marker memorization for one client or adapter isolation across concurrent tenants.
- `tinker_agent/`: train on agent rollouts; see [Train on agent rollouts](#train-on-agent-rollouts).

## Layout

One 8-GPU node, disaggregated (multi-LoRA forbids `--colocate`):

- 4 training GPUs: TP2 for the dense layers, EP4 for the 128 routed experts.
- 4 sampling GPUs: two SGLang engines of 2 GPUs each, serving adapter versions by name.
- 4 adapter slots (`--multi-lora-n-adapters`), rank up to 32, covering attention
  (`linear_qkv`, `linear_proj`), the per-expert MoE projections (`linear_fc1`, `linear_fc2`),
  and the output layer (`output_layer`) so the cookbook's default `train_unembed=True` is servable.

The example enables attention, MLP, and output-head training. Client SDK flags
must match the server's selected groups. See [LoRA target selection](/advanced/lora#hf-target-source-of-truth)
for `--target-modules attn,mlp,unembed`; use `attn,mlp` to disable output-head
training. Tinker accepts only these group names and rejects `--exclude-modules`.

## Run

The gateway implements the `tinker==0.26.2` wire schema (newer SDKs renamed protobuf fields); install that exact version on the serving node and the client:

```bash
pip install "tinker==0.26.2"
```

Start the gateway:

```bash
python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py prepare   # once per node
python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve     # Tinker API on :10613
```

Checkpoints default to `<output_dir>/checkpoints/<run_id>`; use `--save-dir` to choose another root.

The launcher sets `--log-probs-chunk-size 1024`, so log-probs over long sequences are
computed 1024 tokens at a time to keep peak GPU memory down. To change it, add for example
`--extra-args "--log-probs-chunk-size 2048"` to the `serve` command.

Install `tinker` on the client, then run the marker checks:

```bash
# one client: train, save for sampler, sample back the marker
python examples/multi_lora/run_multi_tenant_example.py --base-model /root/models/Qwen3-30B-A3B --mode single

# four tenants training concurrently on the same prompt with different markers;
# passing means the adapters stayed isolated end to end
python examples/multi_lora/run_multi_tenant_example.py --base-model /root/models/Qwen3-30B-A3B --mode multi --clients 4
```

## Train on agent rollouts

To train on what an agent did, for example a coding agent in a sandbox, run the
agent against the Tinker capture proxy with the client package in
[tinker_agent](/examples/multi-lora/tinker-agent). It records each model call's exact tokens
and log-probs and turns an episode into training data.

## Context window

Each sampled sequence, prompt plus completion, must fit one window: the smaller of

- the trainer's per-sequence limit, `--max-tokens-per-gpu` rounded down to a multiple of
  TP × `--data-pad-size-multiplier`, times the context-parallel size; and
- the inference limit, `--sglang-context-length` (the model's maximum length if unset)
  minus 7 tokens, so SGLang still accepts a full-window sequence as a prompt.

With the launcher's defaults the window is 8192 tokens.

When you sample, the gateway:

- lowers `max_tokens` to what is left in the window. The request you sent is unchanged.
- rejects a prompt that leaves no room to generate with HTTP 400 `prompt is too long`,
  before the prompt reaches the engine. Agents can use this error to compact their history.
- still scores a full-window sequence with `compute_logprobs`.

Training datums are checked only against the trainer limit, so a shorter inference window
does not shrink what you can train on.

## Supported inputs

Training accepts text with 1-D loss inputs. 2-D soft targets, including SDFT,
are not supported. Sampling requires a `/sampler_weights/` path returned by
`save_weights_for_sampler()`; `/weights/` training checkpoints cannot be sampled directly.

## Failure handling

A terminal failure of `forward_backward`, `optim_step`, or `load_state` ends
training for that model, including commands already queued behind it.
This includes content validation failures with a valid model and sequence.
Create a new model and restore a saved checkpoint to continue; completed
futures and published checkpoints keep their results.

Known request-local failures of `forward` or sampling leave model training
available. Checkpoint load/save execution failures, including filesystem errors,
invalidate the shared trainer cell and stop the server.
Saving sampler weights commits an immutable directory;
it does not call the inference engines. Sampling loads that snapshot from disk
on demand, including after cache eviction. An engine load failure fails the
sampling request; it leaves the snapshot and training state intact. Unknown
trainer execution failures invalidate the shared trainer cell and stop the server.

This gateway provides failure isolation, not automatic training recovery.
Checkpoints persist; futures, deduplication, and unsaved accumulation do not
survive a server restart.

## Sampler snapshots

Training and inference must use the same base checkpoint. Tinker engines load
that frozen base at startup and serve without trainer weight updates; dummy
loading and `update_weights: true` are rejected. Ordinary full-model and
single-LoRA training continue to use the existing weight updater.

`--tinker-checkpoint-root` must be on storage shared by the trainers, gateway,
and every inference engine. All trainer ranks participate in adapter gathering;
rank 0 writes the tensors and config, then `META.json` after all ranks finish.
Existing sampler versions cannot be overwritten. Saving between `forward_backward`
and `optim_step` neither applies nor discards pending gradients.

Training checkpoint saves and loads are serialized within the gateway. Overwriting
deletes the previous checkpoint before writing the new one; a failed overwrite does
not preserve the previous checkpoint. Checkpoints are ordinary directories;
overwriting does not retain hidden versions.
