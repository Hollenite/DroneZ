# DroneZ Hugging Face Space Deployment

This document is copy-paste oriented for the Space `Krishna2521/dronez-openenv`.

Current status: the Space has been created, but this repo does **not** claim it has been pushed and tested until you run the commands below.

## 1. Space Settings

- Hugging Face Space: `Krishna2521/dronez-openenv`
- Space repo: `https://huggingface.co/spaces/Krishna2521/dronez-openenv`
- Runtime URL to test after build: `https://krishna2521-dronez-openenv.hf.space`
- SDK: `Docker`
- Recommended visibility: `public` if hackathon rules require judge access
- Hardware: CPU Basic is enough for the environment API and replay UI
- Secret/token: use `HF_TOKEN` in your shell if needed; do not paste tokens into docs or chat
- Startup command: handled by the root `Dockerfile`
- Container port: `7860`

## 2. Canonical Dockerfile

Use the root `Dockerfile` for Hugging Face Spaces.

- Root `Dockerfile`: canonical Space build, listens on `PORT=7860`
- `server/Dockerfile`: mirrored OpenEnv-style server layout, kept for compatibility/reference

## 3. Push With Git

```bash
git remote add space https://huggingface.co/spaces/Krishna2521/dronez-openenv
git push space main
```

If the `space` remote already exists:

```bash
git remote set-url space https://huggingface.co/spaces/Krishna2521/dronez-openenv
git push space main
```

## 4. Alternative Push With OpenEnv

```bash
openenv push
```

Use this only after `openenv validate` passes locally.

## 5. Remote Endpoint Checks

```bash
curl https://krishna2521-dronez-openenv.hf.space/health
curl https://krishna2521-dronez-openenv.hf.space/tasks
curl -X POST https://krishna2521-dronez-openenv.hf.space/reset -H "Content-Type: application/json" -d "{}"
curl https://krishna2521-dronez-openenv.hf.space/state
curl -X POST https://krishna2521-dronez-openenv.hf.space/step -H "Content-Type: application/json" -d '{"action":{"action":"hold_fleet","params":{"zone_id":"Z1"}}}'
```

Useful browser links:

- App root: `https://krishna2521-dronez-openenv.hf.space/`
- API docs: `https://krishna2521-dronez-openenv.hf.space/docs`
- Replay UI: `https://krishna2521-dronez-openenv.hf.space/demo/index.html`
- HF repo page: `https://huggingface.co/spaces/Krishna2521/dronez-openenv`

## 6. Troubleshooting

- Build fails on requirements: check the Space logs and confirm the root `requirements.txt` is present.
- Port issue: Hugging Face Docker Spaces require `7860`; the root Dockerfile defaults to `PORT=7860`.
- Space sleeping: open the Space page and wait for rebuild/wakeup before testing curl.
- Auth issue on push: run `hf auth whoami`, or configure Git credentials using a write token locally.
- Token issue: use `HF_TOKEN` environment variable or Hugging Face credential manager; do not put tokens in the repo.
- App works locally but not on HF: compare the local `docker run --rm -p 8000:7860 dronez` test with Space logs.

## 7. Links To Paste Into README

- HF Space: `https://huggingface.co/spaces/Krishna2521/dronez-openenv`
- Runtime app: `https://krishna2521-dronez-openenv.hf.space`
- API docs: `https://krishna2521-dronez-openenv.hf.space/docs`
- Replay UI: `https://krishna2521-dronez-openenv.hf.space/demo/index.html`
