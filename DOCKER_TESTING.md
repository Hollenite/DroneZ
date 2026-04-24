# DroneZ Docker Testing

The root `Dockerfile` is the canonical Hugging Face Space Dockerfile. Hugging Face Docker Spaces expect the app to listen on port `7860`, so the image defaults to `PORT=7860`.

## 1. Start Docker / Colima

```bash
colima start
```

## 2. Build The Image

```bash
docker build -t dronez .
```

## 3. Run With Hugging Face Port Mapping

```bash
docker run --rm -p 8000:7860 dronez
```

Then test from another terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/tasks
curl -X POST http://127.0.0.1:8000/reset -H "Content-Type: application/json" -d "{}"
curl http://127.0.0.1:8000/state
curl -X POST http://127.0.0.1:8000/step -H "Content-Type: application/json" -d '{"action":{"action":"hold_fleet","params":{"zone_id":"Z1"}}}'
```

## 4. Optional Local Port 8000 Inside Container

If you want the container itself to listen on `8000`:

```bash
docker run --rm -e PORT=8000 -p 8000:8000 dronez
```

## 5. Expected Results

- `/health` returns `{"status":"ok"}`
- `/tasks` lists `demo`, `easy`, `hard`, and `medium`
- `/reset` creates or resets the default session
- `/state` returns the current default-session state
- `/step` advances the environment with a valid action
- `/docs` opens FastAPI docs
- `/demo/index.html` opens the trace-driven replay UI when artifacts are present

## 6. Common Failures

- Docker socket missing: run `colima start` or start Docker Desktop.
- Port already in use: change the host side, for example `-p 8010:7860`.
- Slow build: the image installs `openenv-core[core]`, FastAPI, and plotting dependencies.
- Space starts but no app appears: confirm the container listens on `7860`.
