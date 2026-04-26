const $ = (id) => document.getElementById(id);

const taskSelect = $("taskSelect");
const resetBtn = $("resetBtn");
const stateBtn = $("stateBtn");
const validBtn = $("validBtn");
const stepBtn = $("stepBtn");
const observationBox = $("observationBox");
const responseBox = $("responseBox");
const validActions = $("validActions");
const actionEditor = $("actionEditor");
const cumulativeReward = $("cumulativeReward");
const sessionLabel = $("sessionLabel");
const stepLabel = $("stepLabel");
const actionCount = $("actionCount");
const lastReward = $("lastReward");
const doneValue = $("doneValue");
const doneLabel = $("doneLabel");
const reasonValue = $("reasonValue");
const curlBox = $("curlBox");
const pythonBox = $("pythonBox");

const state = {
  sessionId: null,
  observation: null,
  cumulativeReward: 0,
  selectedActionIndex: null,
};

function pretty(payload) {
  return JSON.stringify(payload, null, 2);
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(payload.detail || `${path} failed with HTTP ${response.status}`);
  }
  return payload;
}

function setObservation(observation) {
  state.observation = observation;
  observationBox.textContent = pretty(observation || {});
  stepLabel.textContent = `step ${observation?.step ?? 0}`;
}

function setResponse(payload) {
  responseBox.textContent = pretty(payload || {});
  const reward = Number(payload?.reward ?? 0);
  lastReward.textContent = reward.toFixed(2);
  doneValue.textContent = String(Boolean(payload?.done));
  doneLabel.textContent = payload?.done ? "done" : "not done";
  reasonValue.textContent = payload?.info?.done_reason || "ongoing";
}

function setSession(sessionId) {
  state.sessionId = sessionId;
  sessionLabel.textContent = sessionId ? `session ${sessionId.slice(0, 10)}...` : "No active session";
}

function updateReward(delta = 0, reset = false) {
  state.cumulativeReward = reset ? 0 : state.cumulativeReward + Number(delta || 0);
  cumulativeReward.textContent = state.cumulativeReward.toFixed(2);
}

function fillCodeExamples() {
  const task = taskSelect.value;
  const actionText = actionEditor.value.trim() || "{}";
  curlBox.textContent = [
    `curl -X POST ${location.origin}/reset -H 'Content-Type: application/json' -d '{"task_id":"${task}"}'`,
    "",
    `curl -X POST ${location.origin}/step -H 'Content-Type: application/json' -d '{"action":${actionText}}'`,
  ].join("\n");

  pythonBox.textContent = `import requests

BASE = "${location.origin}"

reset = requests.post(f"{BASE}/reset", json={"task_id": "${task}"}).json()
for step in range(20):
    candidates = requests.post(f"{BASE}/valid-actions", json={}).json()["actions"]
    action = candidates[0]["action"]
    result = requests.post(f"{BASE}/step", json={"action": action}).json()
    print(step, result["reward"], result["done"], result["info"].get("done_reason"))
    if result["done"]:
        break
`;
}

function renderValidActions(actions) {
  validActions.innerHTML = "";
  actionCount.textContent = `${actions.length} candidates`;
  actions.forEach((item) => {
    const card = document.createElement("button");
    card.className = "action-card";
    card.type = "button";
    card.innerHTML = `
      <b>${item.index}. ${escapeHtml(item.action.action)}</b>
      <span>${escapeHtml(item.reason || "Candidate action")}</span>
      <span class="action-meta">
        <i class="pill">priority: ${escapeHtml(item.priority || "normal")}</i>
        <i class="pill">risk: ${escapeHtml(item.risk || "low")}</i>
      </span>
      <code>${escapeHtml(JSON.stringify(item.action))}</code>
    `;
    card.addEventListener("click", () => {
      state.selectedActionIndex = item.index;
      actionEditor.value = pretty(item.action);
      document.querySelectorAll(".action-card").forEach((node) => node.classList.remove("selected"));
      card.classList.add("selected");
      fillCodeExamples();
    });
    validActions.appendChild(card);
  });
  if (actions[0]) {
    actionEditor.value = pretty(actions[0].action);
  }
  fillCodeExamples();
}

async function resetEnvironment() {
  const payload = await request("/reset", {
    method: "POST",
    body: JSON.stringify({ task_id: taskSelect.value }),
  });
  setSession(payload.session_id);
  setObservation(payload.observation);
  setResponse({ info: payload.info, reward: 0, done: false });
  updateReward(0, true);
  await refreshValidActions();
}

async function getState() {
  const payload = await request("/state");
  setSession(payload.session_id);
  setObservation(payload.state?.observation);
  setResponse(payload.state);
  updateReward(0, false);
  await refreshValidActions();
}

async function refreshValidActions() {
  const payload = await request("/valid-actions", {
    method: "POST",
    body: JSON.stringify(state.sessionId ? { session_id: state.sessionId } : {}),
  });
  setSession(payload.session_id || state.sessionId);
  renderValidActions(payload.actions || []);
}

async function stepAction() {
  let action;
  try {
    action = JSON.parse(actionEditor.value);
  } catch (error) {
    responseBox.textContent = `Invalid JSON: ${error.message}`;
    return;
  }
  const payload = await request("/step", {
    method: "POST",
    body: JSON.stringify({ action }),
  });
  setSession(payload.session_id);
  setObservation(payload.observation);
  setResponse(payload);
  updateReward(payload.reward);
  await refreshValidActions();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = $(button.dataset.copy);
    await navigator.clipboard.writeText(target.textContent);
    button.textContent = "Copied";
    setTimeout(() => (button.textContent = "Copy"), 900);
  });
});

resetBtn.addEventListener("click", () => resetEnvironment().catch(showError));
stateBtn.addEventListener("click", () => getState().catch(showError));
validBtn.addEventListener("click", () => refreshValidActions().catch(showError));
stepBtn.addEventListener("click", () => stepAction().catch(showError));
actionEditor.addEventListener("input", fillCodeExamples);

function showError(error) {
  responseBox.textContent = error.stack || error.message || String(error);
}

fillCodeExamples();
resetEnvironment().catch(showError);
