# Hosted tier

The same gateway runs the cloud tier: many customers on one deployment, each billed a share of
what they save. What exists today is the foundation: accounts, tenant-tagged traffic, isolated
stats, metering and invoice calculation. Payments are not wired in (see the last section).

## Turn it on

```yaml
hosted:
  enabled: true
database: postgresql://...      # a shared database for all gateway instances
```

With `hosted.enabled`, every `/v1` call needs a key attached to an account.

```bash
distiller accounts create acme --share 0.3 --price-in 2.50 --price-out 10 --min-monthly 49
distiller keys create acme-prod --account acme              # their app key
distiller keys create acme-admin --account acme --role admin  # their dashboard key
distiller keys create platform --role admin                   # ours: no account = platform admin
```

`--price-in/--price-out` are what the customer's current teacher model costs per 1M tokens; they
set what a student answer is worth. `--student-per-1k` is our own cost to serve student calls,
which is subtracted before we take a share.

## Isolation

- Every logged call carries the account of the key that made it.
- An account admin's `/stats`, `/stats/timeseries` and dashboard show only their own traffic.
- Global actions (switching modes, reload, retrain, task discovery) need a platform admin key.
- `GET /account/usage` returns the account's bill for the month so far; a platform admin passes
  `?account=`.

## Share-of-savings pricing

Per account and calendar month (UTC), per task:

```
teacher cost per call = avg prompt tokens × price in / 1M + avg completion tokens × price out / 1M
avoided  = student-served calls × teacher cost per call
serving  = student-served calls × student_usd_per_1k / 1000
savings  = max(0, avoided − serving)
```

Token averages come from the calls the teacher answered that month (all-time averages if a task
had none). The invoice is `max(min_monthly, share × Σ savings)`: the customer keeps the rest of
the saving, and pays only the minimum in a month where the students saved nothing.

```bash
distiller billing invoice acme --month 2026-10            # show
distiller billing invoice acme --month 2026-10 --save     # store (no silent overwrite)
distiller billing export --out invoices.csv               # one row per task line
```

## Not built yet

- **Payments.** No payment provider is integrated. Saved invoices export to CSV/JSON for an
  invoicing system (Stripe, Chargebee, manual). A provider adapter is the next step.
- **Self-service signup.** Accounts are created by a platform admin with the CLI.
- **Per-account tasks and models.** Tasks, modes and students are shared by the deployment;
  traffic and billing are per account. Per-account task namespaces come next.
