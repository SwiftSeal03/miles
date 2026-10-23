# Train on agent rollouts with Tinker capture

You have an agent that talks to a chat API, such as a coding agent in a sandbox
or a tool-using harness, and you want to train the model on what the agent did.
RL needs the exact tokens the model sampled and their log-probs. Re-tokenizing
the chat transcript afterwards does not give you those back.

`tinker_agent` puts the Tinker cookbook's capture proxy in front of your
sampling client:

1. The agent calls the proxy like any OpenAI- or Anthropic-compatible API.
2. Every model call is recorded with its exact prompt tokens, sampled tokens
   and log-probs.
3. After the episode, you read the calls back as a cookbook `Trajectory` and
   train on it.

It works with the Miles multi-LoRA gateway or the hosted Tinker service, and
the package does not depend on Miles.

## Before you start

- **A Tinker endpoint.** Start the Miles gateway as described in
  [Multi-LoRA Tinker Gateway](../README.md) (it serves on port `10613`), or use
  the hosted Tinker service.
- **The client package**, on Python 3.11 or newer. It pins `tinker==0.26.2` and
  `tinker-cookbook`:

  ```bash
  uv pip install ./examples/multi_lora/tinker_agent
  ```

- **The renderer name for your model**, so that the proxy formats chats with
  the model's own template: `qwen3` for the Qwen3-30B-A3B checkpoint the
  gateway launcher serves, `qwen3_instruct` for Qwen3 Instruct-2507 models. See
  `tinker_cookbook.renderers.get_renderer` for the full list.

## Example

This script runs one two-turn episode through the proxy, scores it, and takes
one PPO step on the captured tokens. The `run_agent` function stands in for your
agent. Save it as `agent_rl.py`:

```python
import argparse
import asyncio
import tempfile

import httpx
import tinker
from tinker_agent.proxy import serve_proxy
from tinker_agent.trajectory import flush_capture, read_samples, rows_to_trajectory
from tinker_cookbook import renderers
from tinker_cookbook.capture.store import capture_to_store
from tinker_cookbook.rl.data_processing import trajectory_to_data

RUN_ID = "demo"


async def run_agent(proxy_url):
    # Stand-in for your agent: any OpenAI-compatible client with base_url=proxy_url + "/v1" works.
    messages = [{"role": "user", "content": "Reply with the number 1."}]
    async with httpx.AsyncClient(base_url=proxy_url, timeout=300) as http:
        for _ in range(2):
            response = await http.post("/v1/chat/completions", json={"messages": messages, "max_tokens": 32})
            response.raise_for_status()
            messages += [response.json()["choices"][0]["message"], {"role": "user", "content": "Now reply with 2."}]
    return 1.0  # the episode's reward, from your verifier


async def main(base_url, base_model, renderer_name, store_url):
    service = tinker.ServiceClient(base_url=base_url, api_key="tml-demo")
    training = await service.create_lora_training_client_async(base_model=base_model, rank=8)
    saved = await (await training.save_weights_for_sampler_async(name="step-0"))
    sampler = await service.create_sampling_client_async(model_path=saved.path)
    renderer = renderers.get_renderer(renderer_name, sampler.get_tokenizer())

    async with serve_proxy(renderer, sampler, model_label=base_model, default_max_tokens=32) as port:
        reward = await run_agent(f"http://127.0.0.1:{port}/r/run/{RUN_ID}/group/0/traj/0")
    await flush_capture()

    rows = await read_samples(store_url, RUN_ID, group_idx=0, traj_idx=0)
    trajectory = rows_to_trajectory(rows, "completed")
    # With several trajectories per task, pass reward minus the group mean instead.
    datums = trajectory_to_data(trajectory, traj_advantage=reward)
    # "mask" is a client-side field; the cookbook trainer also drops it before forward_backward.
    datums = [
        tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={key: value for key, value in datum.loss_fn_inputs.items() if key != "mask"},
        )
        for datum in datums
    ]
    await (await training.forward_backward_async(datums, loss_fn="ppo"))
    await (await training.optim_step_async(tinker.types.AdamParams(learning_rate=1e-5)))
    print(f"trained on {len(rows)} captured calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--renderer", default="qwen3")
    args = parser.parse_args()
    with capture_to_store(RUN_ID, data_dir=tempfile.mkdtemp(prefix="tinker-capture-")) as captured:
        asyncio.run(main(args.base_url, args.base_model, args.renderer, captured.base_url))
```

Run it against the gateway:

```bash
python agent_rl.py --base-url http://<gateway-host>:10613 --base-model /root/models/Qwen3-30B-A3B
```

For a full recipe that runs Harbor tasks in sandboxes and trains with the
cookbook's RL loop, see [Harbor recipe](../harbor_tinker/README.md).

## How it works

1. **Capture.** `capture_to_store(run_id, data_dir=...)` starts a local capture
   store and records every Tinker sampling call made inside it. Wrap the whole
   job in it.
2. **Proxy.** `serve_proxy(renderer, sampling_client, model_label=...,
   default_max_tokens=...)` serves the proxy on a free local port and yields
   the port. `default_max_tokens` applies when the agent's request does not set
   `max_tokens`.
3. **Agent URL.** Give each episode its own URL:
   `http://<host>:<port>/r/run/<run_id>/group/<group_idx>/traj/<traj_idx>`.
   - `run_id` must match the `capture_to_store` run.
   - A group is the set of episodes you compare with each other, for example
     several attempts at one task. The `group/<group_idx>` segment is optional.
   - A traj is one episode.
   - OpenAI clients use this URL plus `/v1` as their base URL (for
     `/v1/chat/completions`). Anthropic clients use the URL itself (for
     `/v1/messages`).
4. **Flush.** After the episodes finish, leave the `serve_proxy` block so
   in-flight calls complete, then call `flush_capture()`. It raises if any call
   failed to be recorded, so you never train on a partial episode.
5. **Read.** `read_samples(store_url, run_id, group_idx=..., traj_idx=...)`
   returns an episode's calls in the order they were made.
   `rows_to_trajectory(rows, stop_reason)` turns them into a `Trajectory`
   without re-tokenizing. `stop_reason` is how the episode ended, such as
   `"completed"`.
6. **Train.** `trajectory_to_data(trajectory, traj_advantage)` builds the
   Datums.
7. **Update the policy.** After `optim_step`, save new sampler weights and serve
   the next episodes through a new proxy with the new sampling client.

## Agents in a sandbox

By default the proxy listens on `127.0.0.1`. To accept connections from a
sandbox, pass `host="0.0.0.0"` and an `auth_token`. The proxy refuses a
non-loopback address without a token. Give the agent the token as its API key:

- OpenAI clients send it as `Authorization: Bearer <token>`.
- Anthropic clients send it as `x-api-key: <token>`.

Advertise an address the sandbox can reach, not `0.0.0.0`.

## What you handle

- **Rewards.** Captured calls carry no reward. Compute it from your verifier and
  pass it as the advantage.
- **Failed calls.** Calls that fail are not retried, and they leave no row, so
  the episode's trajectory contains only the calls that succeeded. Your agent
  decides whether to retry or give up.
- **Long histories.** When a prompt no longer fits the gateway's context window,
  the request fails with HTTP 400 "prompt is too long". The agent should compact
  its history or end the episode. See
  [Context window](../README.md#context-window).
- **Several attempts.** The proxy rejects `n` other than 1. Run each attempt as its own
  episode, with its own `traj_idx` in the same group.

The package adds no retries, refusal handling, request rewriting or
partial-episode recovery.
