# ShopSimulator

This is the embedded ShopSimulator environment used by the tutorial. The
snapshot includes the product archive, Environment v2.1, Reward v3 and the
structured `/api/shop_agent` service. Its upstream source commit is recorded in
[`EMBEDDED_SOURCE.json`](EMBEDDED_SOURCE.json).

Users should not install or start this directory manually. From the repository
root, run:

```bash
bash scripts/setup.sh --shopsim-only
bash scripts/start_environment.sh
```

The standalone setup uses `uv` to create an isolated Python 3.10 environment,
install the pinned CPU service dependencies, verify/extract the embedded product
data and build the SQLite FTS5 index. It does not install training dependencies
or patch an existing veRL checkout. Set `SHOPSIM_PYTHON` to use an existing Python
executable. Omit `--shopsim-only` only when installing the full training stack.

The service runs in the foreground on port 5700 by default. Set `SHOPSIM_PORT`
or `SHOPSIM_ENV_SLOTS` to change the port or concurrent environment slots; slots
are independent of the NPU count. Console output is appended to
`outputs/shopsimulator/server.log` at the repository root. No model server, Ray
cluster or accelerator is needed to start this service.

Generated product JSON, search indexes, virtual environments and logs are not
committed.
