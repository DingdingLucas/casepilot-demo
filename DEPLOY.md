# Deploying the demo to a real URL

The app is deploy-ready: gunicorn entrypoint, `render.yaml` blueprint, optional
password gate (`DEMO_PASSWORD`), and a cache warmer (`WARM_CACHE=1`) so visitors
never hit the ~45s cold start.

## One-time steps (require your accounts)

1. **Push this folder to a GitHub repo** (private is fine).
2. **Render** (render.com) → New → Blueprint → select the repo.
   Render reads `render.yaml` and creates the service. When prompted, set:
   - `ANTHROPIC_API_KEY` — use a **fresh key** (rotate the old one)
   - `DEMO_PASSWORD` — the password you'll share with investors
3. You immediately get a stable URL like `https://casepilot-demo.onrender.com`.

## Custom domain

1. Buy a domain at any registrar (~$10/yr). As of 2026-07-18, `casepilot.com`,
   `casepilot.ai`, `casepilot.io`, `getcasepilot.com` are TAKEN;
   `trycasepilot.com` was available.
2. Render dashboard → your service → Settings → Custom Domains → add
   `demo.yourdomain.com`.
3. At your registrar, add the CNAME record Render shows you
   (`demo` → `casepilot-demo.onrender.com`). HTTPS is automatic.

## Local dev (unchanged)

```
ANTHROPIC_API_KEY=$(cat .apikey) venv/bin/python app.py   # http://localhost:5001
```

Password and warmer are off locally unless you set `DEMO_PASSWORD` / `WARM_CACHE=1`.

## Production test (what the server runs)

```
ANTHROPIC_API_KEY=$(cat .apikey) DEMO_PASSWORD=test PORT=5002 \
  venv/bin/gunicorn --workers 1 --threads 8 --timeout 300 --bind 0.0.0.0:$PORT app:app
```
