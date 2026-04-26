# DroneZ Delivery Simulator — Setup Guide

## Quick Start (Any OS)

### Step 1: Clone the repo
```bash
git clone https://github.com/Hollenite/DroneZ.git
cd DroneZ
```

### Step 2: Start local server

**macOS / Linux:**
```bash
python3 -m http.server 8091
```

**Windows (Python installed):**
```bash
python -m http.server 8091
```

**Windows (no Python — use Node.js):**
```bash
npx -y serve . -l 8091
```

**Windows (no Python, no Node — use PowerShell):**
```powershell
# Download Python from https://python.org/downloads first, then:
python -m http.server 8091
```

### Step 3: Open in browser
```
http://127.0.0.1:8091/dronez_simulator3/index.html
```

Or simply:
```
http://localhost:8091/dronez_simulator3/index.html
```

---

## Controls

| Button | Action |
|--------|--------|
| ▶ Play | Start simulation |
| ⏸ Pause | Pause simulation |
| ↻ Reset | Reset all missions |
| 🚀 Launch | Launch full fleet |
| ☀ Day / 🌙 Night | Toggle theme |
| Speed slider | 0.3x – 3x speed |

### Camera Modes (dropdown)
| Mode | View |
|------|------|
| 🏗 Control Tower | Wide overview |
| 🎯 Follow Drone | Tracks selected drone |
| 🚁 Chase View | Behind drone |
| 👁 First Person | Drone's perspective |
| 🎬 Cinematic | Smooth auto-orbit |
| 📡 Top-Down | Bird's eye |

### Mode Buttons (bottom bar)
- **Operations** — Full dashboard with tower view
- **Follow** — Camera follows selected drone
- **Cinematic** — Presentation mode (film bars)
- **Warehouse** — Focus on drone port/dock
- **Lifecycle Demo** — Follow active delivery mission

### Drone Selection
- Use dropdown in side panel
- Use ◀ Prev / Next ▶ buttons
- Switch between 42 drones

---

## What You'll See

6 drones run **simultaneous delivery missions** with different packages:
- 🏥 Medical Kit → Hospital District
- 🍱 Food Parcel → Campus
- 📦 Electronics → Suburb
- 🚨 Emergency Supply → Market Zone
- 📄 Documents → Emergency Landing
- 🛒 Grocery Package → Cold Chain Storage

Each drone follows the full lifecycle:
```
Order Received → Drone Assigned → Launch → To Pickup → Collect Package →
To Delivery → Deliver (landing/winch) → Return to Warehouse → Dock & Charge
```

---

## Troubleshooting

**"Module not found" error:**
You must run from the DroneZ root folder, not from inside dronez_simulator3/.

**Page is blank / boot screen stuck:**
- Check browser console (F12) for errors
- Make sure you're using Chrome, Edge, or Firefox (Safari may have issues with ES modules)

**Laggy / choppy:**
- Close other heavy tabs/apps
- The simulation is already optimized (no bloom, no shadows, pixel ratio 1)
- If still laggy, reduce speed to 0.3x

**Port already in use:**
```bash
# Use a different port
python3 -m http.server 9090
# Then open: http://localhost:9090/dronez_simulator3/index.html
```

---

## Requirements

- **Browser:** Chrome 90+, Firefox 90+, or Edge 90+ (WebGL2 required)
- **Server:** Python 3.x OR Node.js OR any static file server
- **No npm install needed** — pure HTML/JS/CSS, Three.js loaded from CDN
