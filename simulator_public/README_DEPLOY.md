# DroneZ 3D Simulator Static Bundle

This folder is the deployment-ready static bundle for the DroneZ 3D simulator.

## Files

- `index.html`
- `styles.css`
- `app.js`
- `world.js`
- `missions.js`
- `traces.js`

The simulator loads Three.js from CDN and does not require a backend.

## Local Preview

```bash
cd simulator_public
python -m http.server 8091
```

Open:

```text
http://127.0.0.1:8091
```

## Deploy

Use this folder as a static site on Vercel or Netlify.

Vercel:

```text
Framework Preset: Other
Root Directory: simulator_public
Build Command: empty
Output Directory: .
```

Netlify:

```text
Build command: empty
Publish directory: simulator_public
```
