# DroneZ 3D Simulator Static Deployment

This guide is for the standalone 3D simulator only.

Official DroneZ Hugging Face Space:

```text
https://huggingface.co/spaces/Krishna2521/dronez-openenv
```

## What The Simulator Is

The 3D simulator is a static browser app. It uses:

- `index.html`
- `styles.css`
- `app.js`
- `world.js`
- `missions.js`
- `traces.js`
- Three.js from CDN through an import map

It does not need FastAPI, Python, Docker, model files, or training artifacts.

## Static Deployment Folder

Use this folder for hosting:

```text
simulator_public/
```

It contains only the static simulator files required for deployment.

## Local Preview

From the repo root:

```bash
python -m http.server 8091
```

Open:

```text
http://127.0.0.1:8091/dronez_simulator3/index.html
```

To preview the deployment folder directly:

```bash
cd simulator_public
python -m http.server 8091
```

Open:

```text
http://127.0.0.1:8091
```

## Vercel Settings

Vercel failed before because it detected the full repo as a Python/FastAPI backend. This simulator is static, so Vercel must be pointed at the static simulator output instead.

Recommended Vercel project settings:

```text
Framework Preset: Other
Root Directory: repository root
Install Command: empty
Build Command: empty
Output Directory: simulator_public
```

This repo also includes `vercel.json`:

```json
{
  "version": 2,
  "installCommand": null,
  "buildCommand": null,
  "outputDirectory": "simulator_public"
}
```

Alternative Vercel setup:

```text
Root Directory: simulator_public
Framework Preset: Other
Install Command: empty
Build Command: empty
Output Directory: .
```

If Vercel still tries to detect FastAPI, create a tiny separate GitHub repo containing only the files from `simulator_public/` and deploy that repo as a static site.

## Netlify Settings

Netlify is often the easiest path for static HTML/CSS/JS.

Recommended Netlify settings:

```text
Build command: leave empty
Publish directory: simulator_public
```

This repo also includes `netlify.toml`:

```toml
[build]
  publish = "simulator_public"
```

Manual Netlify fallback:

1. Open Netlify.
2. Use "Add new site" -> "Deploy manually".
3. Drag and drop the `simulator_public/` folder.
4. Copy the generated public URL.
5. Add that URL to the production UI after testing it.

## After Deployment

Current public simulator URL:

```text
https://creative-klepon-8a1f8a.netlify.app/
```

If you replace or rename the public deployment later:

1. Test it in a browser.
2. Confirm it opens in bright/day mode by default.
3. Confirm the night toggle still works.
4. Update the production UI and README with the new URL.
