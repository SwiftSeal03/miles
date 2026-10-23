"""MiMo-V2.6-Flash-RL multi-LoRA training served through the Tinker gateway (`serve_tinker.py`).

Tinker clients train their own LoRA adapters through explicit forward_backward / optim_step calls and sample
from saved adapter versions (see examples/multi_lora/); the server owns no dataset. The adapters cover the
attention, the MLP (dense layer 0 and the routed experts) and the output head of the BF16 checkpoint that
scripts/run_mimo_v2_6_flash.py trains in full; `prepare` runs the same conversion.

Multi-LoRA keeps every adapter's accumulated gradients on the training GPUs between calls, so training and
sampling use separate GPUs, with PP 1, no offload, and the whole BF16 base resident on both sides:
  mimo26-p4-bf16           4 training GPUs (TP4, EP4) + one TP4 engine: one 8-GPU node
  MiMo-V2.6-Flash-RL-bf16  16 training GPUs (TP4, EP16) + one TP8 engine: three 8-GPU nodes. The base alone
                           takes 8 training GPUs (EP8), where the expert adapters of 4 slots at rank 32 no
                           longer fit; EP8 leaves room for roughly n_adapters * lora_rank <= 64.
Sampling also needs SGLang LoRA support for MiMo-V2: per-layer qkv widths, a value head narrower than the key
head, the dense first layer, and skipping the vision/audio towers.

Args:
  --model-name: checkpoint directory under `--model-dir`; selects the layout above.
  --actor-num-gpus, --tensor-model-parallel-size, --expert-model-parallel-size: training GPUs and their TP and
      EP; None takes the layout of --model-name. TP must not exceed 4, the global-attention KV head count.
  --n-adapters, --lora-rank, --lora-alpha: adapter slots, their maximum rank (each client picks a rank up to
      it), and the LoRA alpha shared by every slot.
  --target-modules: LoRA groups among attn, mlp and unembed; clients must request the same groups.
  --tinker-port: port of the Tinker API on the head node.
  --extra-args: appended verbatim to the serve_tinker.py argv.

The full model spans three nodes: start the ray head on the first, join the others, and run with
`MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=<head ip> --num-nodes 3`.

Examples:
  python scripts/run_mimo_v2_6_flash_tinker.py prepare
  python scripts/run_mimo_v2_6_flash_tinker.py serve --n-adapters 4 --lora-rank 32
"""

from dataclasses import dataclass, field
from typing import Literal

import typer
from scripts.run_mimo_v2_6_flash import _RECIPES, _prepare_checkpoint

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass(frozen=True)
class _Layout:
    actor_num_gpus: int
    tensor_model_parallel_size: int
    expert_model_parallel_size: int


_LAYOUTS = {
    "MiMo-V2.6-Flash-RL-bf16": _Layout(actor_num_gpus=16, tensor_model_parallel_size=4, expert_model_parallel_size=16),
    "mimo26-p4-bf16": _Layout(actor_num_gpus=4, tensor_model_parallel_size=4, expert_model_parallel_size=4),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    model_name: Literal["MiMo-V2.6-Flash-RL-bf16", "mimo26-p4-bf16"] = "mimo26-p4-bf16"
    num_gpus_per_node: int = 8
    # None takes the layout of --model-name.
    actor_num_gpus: int | None = None
    tensor_model_parallel_size: int | None = None
    expert_model_parallel_size: int | None = None
    n_adapters: int = 4
    lora_rank: int = 32
    lora_alpha: int = 64
    target_modules: str = "attn,mlp,unembed"
    tinker_port: int = 10613
    max_tokens_per_gpu: int = 9216
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""

    def __post_init__(self):
        layout = _LAYOUTS[self.model_name]
        self.actor_num_gpus = self.actor_num_gpus or layout.actor_num_gpus
        self.tensor_model_parallel_size = self.tensor_model_parallel_size or layout.tensor_model_parallel_size
        self.expert_model_parallel_size = self.expert_model_parallel_size or layout.expert_model_parallel_size

    @property
    def actor_num_nodes(self) -> int:
        nodes = max(1, self.actor_num_gpus // self.num_gpus_per_node)
        assert self.actor_num_gpus % nodes == 0, f"{self.actor_num_gpus} training GPUs do not fill {nodes} nodes"
        return nodes


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Convert the official checkpoint to BF16 under --model-dir. Run once per node before serving."""
    _prepare_checkpoint(args.create_backend(), model_dir=args.model_dir, model_name=args.model_name)


@app.command()
@U.dataclass_cli
def serve(args: ScriptArgs):
    """Serve the Tinker gateway (idles until clients connect)."""
    recipe = _RECIPES[args.model_name]
    tp = args.tensor_model_parallel_size

    ckpt_args = f"--hf-checkpoint {args.model_dir}/{args.model_name} --megatron-to-hf-mode bridge "

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout 0.0 "
        "--no-gradient-accumulation-fusion "
        f"--multi-lora-n-adapters {args.n_adapters} "
        f'--target-modules "{args.target_modules}" '
    )

    tinker_args = (
        f"--tinker-server-port {args.tinker_port} "
        f"--tinker-checkpoint-root {args.output_dir}/checkpoints/{args.run_id} "
    )

    # initial values only; every optim_step request carries its own Adam parameters
    optimizer_args = "--optimizer adam --lr 1e-4 --accumulate-allreduce-grads-in-fp32 "

    perf_args = (
        f"--tensor-model-parallel-size {tp} "
        f"{'--sequence-parallel ' if tp > 1 else ''}"
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.expert_model_parallel_size} "
        "--expert-tensor-parallel-size 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    )

    # one engine holding the whole BF16 base
    sglang_args = (
        f"--rollout-num-gpus {recipe.rollout_num_gpus_per_engine} "
        f"--rollout-num-gpus-per-engine {recipe.rollout_num_gpus_per_engine} "
        "--sglang-dtype bfloat16 "
        f"--sglang-mem-fraction-static {recipe.sglang_mem_fraction_static} "
        "--sglang-lora-backend triton "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--attention-backend fused "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.actor_num_gpus // args.actor_num_nodes} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    train_args = (
        f"{ckpt_args} {lora_args} {tinker_args} {optimizer_args} {perf_args} {sglang_args} {misc_args} "
        f"{args.extra_args} "
    )

    args.create_backend().execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=recipe.megatron_model_type,
        megatron_path=args.megatron_path,
        train_script="serve_tinker.py",
        job_lifetime="launcher",
    )


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
