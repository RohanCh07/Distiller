# Deployment

## One machine

```bash
distiller gateway --host 0.0.0.0 --workers 4
```

Each worker loads its own students (memory × workers). All workers share the SQLite file (WAL
mode, 30 s busy timeout): modes, keys, logs, drift state and retrain jobs are in the database.

## Several machines: Postgres

```yaml
database: postgresql://distiller:secret@db.internal:5432/distiller
```

Install `distiller[postgres]`. The schema is created on start-up; columns added in later
versions are added automatically. Every store test runs against both SQLite and a real Postgres.

What makes several gateways safe:

- Mode changes, API keys and revocations are read from the database on each request.
- Drift demotion is a conditional update (`serve → shadow` only if still `serve`), so when
  several instances see the same drift at once, exactly one demotes and one event is recorded.
- Retrain jobs are claimed with a conditional update, so two workers never run the same job.
- New models: `runs/<task>/best.json` is replaced atomically; each gateway notices the change and
  reloads. Put `runs/` on a shared volume, or call `POST /admin/tasks/<task>/reload` on each.

## Docker

`deploy/docker-compose.yml` runs Postgres, two gateway replicas behind nginx (with streaming
buffering off) and an optional vLLM service (`--profile gpu`):

```bash
cp distiller.example.yaml deploy/config/distiller.yaml   # set database: postgresql://...@postgres/...
docker compose -f deploy/docker-compose.yml up --build
```

CI builds `deploy/Dockerfile` on every push.

## Checklist

- `distiller doctor` passes with no problems.
- `security.require_api_key: true` and an admin key exist.
- `scrub_pii` and `encrypt_logs` set to what your data policy needs.
- The teacher key comes from the environment (`teacher.api_key_env`), never from the config file.
