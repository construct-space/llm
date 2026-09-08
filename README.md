# tinker-proxy

OpenAI-compatible HTTP shim fronting [Tinker](https://tinker.thinkingmachines.ai/)'s
sampling API. Surface for Construct's Source fine-tunes.

Single Python process, no DB, no state. One sampling client per configured
model, lazy-init on first request.

Deployed publicly at **`https://llm.lisaos.dev/`** (Caprover, Let's
Encrypt TLS auto-managed). Internal callers (`api/provider`) hit it by
the same URL or via the Caprover internal hostname `http://tinker-proxy/`
(port 80 maps to container 11435).

## What it serves

- `GET  /health`               — unauthed; Caprover health check
- `GET  /v1/models`            — auth required; lists configured samplers
- `POST /v1/chat/completions`  — auth required; non-streaming OpenAI chat

Routes by request `model` field → Tinker sampler. `model` must match a key
in the `SAMPLERS` env map (e.g. `source-medium`, `source-vision`).

## Auth

Bearer tokens, set via `API_KEYS` env var (comma-separated). Generate
with `openssl rand -hex 32`. The same value(s) go into
`construct_upstream_providers.api_key_encrypted` so `api/provider` can
authenticate when it dispatches Source chats.

**If `API_KEYS` is empty, `/v1/*` is OPEN.** Only safe for a private
Caprover network. Never deploy publicly without keys — your single
`TINKER_API_KEY` quota is unguarded.

## Caprover deploy (public)

1. **App name: `llm`** (folder is `api/tinker/` for the codebase, but the
   Caprover app and public domain are `llm` / `llm.lisaos.dev`).

2. **HTTP settings**:
   - Container HTTP Port = `11435`
   - Enable HTTPS, force HTTPS redirect
   - Connect domain `llm.lisaos.dev` (Caprover handles cert via Let's Encrypt)

3. **Env vars** — paste from `.env.sample` and fill in:
   - `TINKER_API_KEY` (from your `~/.tinker/env`)
   - `SAMPLERS` (JSON map; sampler paths from `tinker checkpoint list`)
   - `API_KEYS` (one or more bearer tokens, comma-separated)

4. **Deploy** — from this directory:
   ```bash
   caprover deploy
   ```
   Or wire to a git remote that Caprover tracks for auto-deploy.

5. **Verify**:
   ```bash
   curl https://llm.lisaos.dev/health
   # {"status":"ok","models":["source-medium","source-vision"]}

   curl https://llm.lisaos.dev/v1/models \
     -H "Authorization: Bearer $API_KEY"
   # {"object":"list","data":[…]}

   curl -X POST https://llm.lisaos.dev/v1/chat/completions \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"source-medium","messages":[{"role":"user","content":"hi"}]}'
   ```

## Pairing with `api/provider`

After deploy, in Adminer (or Oracle UI):

```sql
UPDATE construct_upstream_providers
SET base_url = 'https://llm.lisaos.dev/v1',
    enabled  = true
WHERE id = 'tinker';

-- and paste one of API_KEYS as the api_key_encrypted via Oracle's
-- provider/upstreams page (it encrypts on save).

UPDATE construct_routing_targets
SET enabled = true
WHERE id IN ('tinker-source-medium', 'tinker-source-vision');
```

`construct_picker_entries` rows (`source-medium`, `source-vision`) are
already enabled by the seed.

## Adding a new sampler

Append to `SAMPLERS` JSON; redeploy. Then add a routing target + picker
entry in `api/provider/seeds/tinker.sql`.

## Local dev

```bash
cp .env.sample .env
# fill in TINKER_API_KEY + SAMPLERS; leave API_KEYS empty for unauthed local
set -a; . ./.env; set +a
pip install -r requirements.txt
python main.py
```

Then `curl http://localhost:11435/v1/models`.

## Future

- **Streaming responses** — needs SSE; `api/inference/` does this in Go,
  port the pattern.
- **Per-key rate limits / quotas** — currently bearer tokens are uniform.
  Real product use needs per-tenant accounting.
- **Public provider catalog entry** — if Source becomes a BYO-key product,
  add a `provider_catalog` row pointing at `https://llm.lisaos.dev/v1`.
