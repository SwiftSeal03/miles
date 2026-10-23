# Agent rollouts through official Tinker capture

`tinker_agent` turns agent sessions into cookbook training data using only the
official Tinker cookbook capture stack: the pinned `tinker_cookbook.capture.proxy`,
SDK instrumentation, exporter and capture store.

- It is an independently installable client package and imports no Miles code.
- It accepts a sampling client from either the Miles gateway or the official
  Tinker service.

```bash
uv pip install ./examples/multi_lora/tinker_agent
```

## Recipe

1. Run the recipe inside the official `capture_to_store(run_id, data_dir=...)`.
2. Pass the sampling client directly to `serve_proxy`.
3. Give each agent its own proxy URL, with
   `/r/run/<run_id>/group/<group_idx>/traj/<traj_idx>` before `/v1/...`.
4. After the group's trials finish, close the proxy so aiohttp finishes its
   accepted handlers, then call `flush_capture()`.
5. Read each trajectory's rows with `read_samples()` and convert them with
   `rows_to_trajectory()`.

```python
from tinker_agent.proxy import serve_proxy
from tinker_agent.trajectory import flush_capture, read_samples, rows_to_trajectory
from tinker_cookbook.capture.store import capture_to_store

with capture_to_store(run_id, data_dir=capture_dir) as captured:
    ...
    async with serve_proxy(renderer, sampling_client, model_label="policy", default_max_tokens=8192) as port:
        url = f"http://{host}:{port}/r/run/{run_id}/group/{group_idx}/traj/{traj_idx}"
        ...  # run the agents against their URLs
    await flush_capture()
    rows = await read_samples(captured.base_url, run_id, group_idx=group_idx, traj_idx=traj_idx)
    trajectory = rows_to_trajectory(rows, stop_reason)
```

For sandbox callers, pass `serve_proxy` a reachable `host` and an `auth_token`,
following the official proxy's bind policy.

## API

| Function | What it does |
|---|---|
| `serve_proxy(renderer, sampling_client, *, model_label, default_max_tokens, host, auth_token)` | Serves the official proxy app on a free port and yields the port. |
| `flush_capture()` | Flushes the official exporter and checks its loss counters. A reported capture loss fails training rather than silently consuming an incomplete dataset. |
| `read_samples(store_url, run_id, **coordinates)` | Queries the store's public, paginated rows API by run, group and trajectory, and sorts the rows by official call timestamps. |
| `rows_to_trajectory(rows, stop_reason)` | Converts the rows to cookbook transitions without rebuilding tokens. |

The `flush_capture()` check is a training-consumer check, not a change to
official capture or request behavior.

## Who owns what

| Concern | Owner |
|---|---|
| Rendering, tools, stop semantics, SSE | Official proxy |
| Capture | Official SDK instrumentation |
| Sampled-token record | Official capture store (the only one) |
| Rewards and trial outcomes | Harness |
| Which exact token-prefix continuations share a Datum | Cookbook |
| Failed requests | Agent/harness; they remain in official capture annotations |

There is no separate recorder, token schema or episode state machine. The
recipe does not add sampling retries, refusal matching, parameter overrides or
partial-episode recovery.

## Tests

- **Isolated tests:** exercise the real official proxy → instrumentation →
  exporter → store path with a fake sampler, and validate store-row conversion
  to training data.
- **GPU acceptance:** trains from actual rows in the official store, then
  samples from the updated policy.
