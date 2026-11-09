# Security

```yaml
security:
  require_api_key: true      # /v1 needs a Distiller key
  scrub_pii: true            # remove personal data before anything is stored
  encrypt_logs: true         # encrypt stored content (needs distiller[security])
```

## API keys

```bash
distiller keys create web-app                 # role app: may call /v1
distiller keys create ops --role admin        # also /admin, /stats, the dashboard data
distiller keys list
distiller keys revoke web-app
```

Keys look like `dk-...`, are printed once, and only a SHA-256 hash is stored. Apps send them in
one of two ways:

- `X-Distiller-Key: dk-...`, keeping their own `Authorization` for the teacher; or
- `Authorization: Bearer dk-...`, i.e. just `OPENAI_API_KEY=dk-...` in an OpenAI SDK. The
  gateway then uses its own teacher key upstream. A Distiller key is never forwarded.

`DISTILLER_API_KEYS` (comma-separated) and `DISTILLER_ADMIN_KEY` add keys from the environment,
useful for bootstrapping. Admin endpoints require an admin key as soon as one exists (or when
`require_api_key` is on). A key that is sent but unknown is always rejected with 401.

## PII scrubbing

With `scrub_pii`, these are replaced before storage (inputs, request messages, outputs):

| Kind | Example | Check |
|---|---|---|
| email | `a.b@example.com` → `<EMAIL>` | pattern |
| card | `4111 1111 1111 1111` → `<CARD>` | Luhn |
| aadhaar | `2341 2341 2346` → `<AADHAAR>` | Verhoeff |
| phone | `+91 98765 43210`, `+44 20 7946 0958` → `<PHONE>` | pattern |
| pan | `ABCPE1234F` → `<PAN>` | pattern |
| iban | `GB82 WEST 1234 5698 7654 32` → `<IBAN>` | mod 97 |
| ip | `192.168.1.20` → `<IP>` | pattern |

Checksums keep long reference numbers (UPI refs, order ids) from being mistaken for cards.
UPI handles (`swiggy@ybl`) are kept on purpose: they usually name the merchant. Pick kinds with
`security.pii_kinds`.

The teacher still receives the original text (it needs it to answer). The **student** is given
the scrubbed text, the same as the training data it learned from, so training and serving inputs
match.

## Encryption at rest

```bash
export DISTILLER_ENCRYPTION_KEY=$(distiller keys encryption-key)
```

Stored content columns are encrypted with Fernet (AES-128-CBC + HMAC-SHA256). Counts, timings and
agreement stay readable, so stats and the dashboard work without decrypting. Export and retraining
decrypt transparently.

Rotation: put the new key first, the old one second
(`DISTILLER_ENCRYPTION_KEY=new,old`), run `distiller keys rotate-encryption`, then drop the old
key. Rows written before encryption was enabled are read as-is.

!!! warning
    Losing the encryption key makes the stored content unreadable. Keep it in your secret
    manager, not in `distiller.yaml`.
