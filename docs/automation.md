# Automation

## Task discovery

Untagged calls are logged as `unassigned:<hash of system prompt>`, so each group is one repeated
prompt.

```bash
distiller task discover --min-calls 200                 # list candidates
distiller task discover --write tasks/ --adopt          # write task.yaml + data, adopt the calls
```

For each group the teacher's answers decide the task type:

| Answers look like | Proposed task |
|---|---|
| JSON objects (90%+) | `extraction`, with a JSON schema inferred from them (required keys, types, enums for repeated categories) |
| a few distinct short answers, each repeated | `classification`, with those labels |
| short free text | `generation` |
| long free-form text | not proposed (a small model is a poor fit) |

`--adopt` moves the group's logged calls to the new task name; add the printed
`match_system_prompt` line to `distiller.yaml` so future calls land there too. The dashboard
lists candidates as well (`GET /admin/discover`).

## Retraining on drift

```yaml
retrain:
  enabled: true
  min_new_examples: 100
  reload_seconds: 30
tasks:
  banking77:
    mode: serve
    model_dir: best
    task_file: examples/banking77/task.yaml
```

1. In `serve` mode, `audit_rate` of student answers are re-checked by the teacher.
2. If audited agreement drops below `demote.min_agreement`, the task goes back to `shadow` and a
   retrain job is queued. In shadow the teacher answers everything, and those answers are logged:
   that is the fresh training data.
3. A worker, on any machine that can train:

    ```bash
    distiller retrain-worker            # loops; --once for cron
    ```

    trains on the original data plus everything logged for the task, with the usual
    validation/calibration/threshold. The new model is published (`best.json` replaced atomically)
    **only if it is promotable**, meaning some confident slice reaches the target accuracy on a test set
    that now includes recent traffic. Otherwise the current model stays.
4. Gateways notice the new `best.json` and reload it. A broken model never replaces a working one.
   Drift and promotion windows restart for the new model.

Queue a retrain by hand with `distiller retrain <task>` or `POST /admin/tasks/<task>/retrain`.

## Auto-promotion

```yaml
promote:
  enabled: true
  min_agreement: 0.97
  min_samples: 200
```

A task in `shadow` moves to `serve` by itself once its shadow agreement over the last `window`
comparisons (since the last mode change) is at least `min_agreement` on at least `min_samples`.
Leave it off if a person should approve every switch.
