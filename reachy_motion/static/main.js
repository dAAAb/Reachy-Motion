// Reachy-Motion settings page: start/stop modes, show the conversation, gestures and a live pose preview.
const $ = (s) => document.querySelector(s);
let after = 0;
let robotLine = null;

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  return r.json();
}

document.querySelectorAll(".modes button[data-mode]").forEach((b) =>
  b.addEventListener("click", () => api("/api/start", { mode: b.dataset.mode })));
$("#stop").addEventListener("click", () => api("/api/stop", {}));

function addEvent(cls, tag, text, extra) {
  const li = document.createElement("li");
  li.className = cls;
  const t = document.createElement("span");
  t.className = "tag";
  t.textContent = tag;
  li.append(t, document.createTextNode(" " + text));
  if (extra) {
    const r = document.createElement("span");
    r.className = "recipe";
    r.textContent = extra;
    li.append(r);
  }
  const list = $("#events");
  list.append(li);
  while (list.children.length > 200) list.firstChild.remove();
  list.scrollTop = list.scrollHeight;
  return li;
}

function onEvent(e) {
  switch (e.type) {
    case "status": $("#status").textContent = (e.mode ? e.mode + " · " : "") + e.text; break;
    case "user": addEvent("ev-user", "you", e.text); robotLine = null; break;
    case "robot_delta":
      if (!robotLine) robotLine = addEvent("ev-robot", "reachy", "");
      robotLine.lastChild.textContent += e.delta === undefined ? e.text : e.delta;
      break;
    case "robot": if (!robotLine) addEvent("ev-robot", "reachy", e.text); robotLine = null; break;
    case "gesture":
      addEvent("ev-gesture " + e.source, e.source === "planner" ? "gesture" : e.source,
        `${e.idea}${e.latency_ms ? ` · ${e.latency_ms} ms` : ""}`, e.recipe);
      break;
    case "timing": addEvent("ev-timing", "timing", `ASR ${e.asr_s}s · first audio ${e.first_audio_s}s`); break;
    case "interrupt": addEvent("ev-int", "barge-in", "— interrupted"); robotLine = null; break;
    case "command": addEvent("ev-cmd", "command", `${e.kind} → ${e.arg}`, e.text); break;
    case "seen": addEvent("ev-cmd", "camera", e.text, e.question); break;
    case "found": addEvent("ev-cmd", "web", e.text, e.question); break;
  }
}

// pose = [x, y, z, roll, pitch, yaw, antenna_right, antenna_left, body_yaw] (m / rad)
function drawPose(p) {
  if (!p) return;
  const deg = (r) => (r * 180) / Math.PI;
  const [, , z, roll, pitch, yaw, ar, al, body] = p;
  const dx = Math.sin(yaw) * 46, dy = -z * 1600 + Math.sin(pitch) * 60;
  $("#head").setAttribute("transform", `translate(${dx.toFixed(1)} ${dy.toFixed(1)}) rotate(${(-deg(roll)).toFixed(1)})`);
  $("#antR").setAttribute("transform", `rotate(${deg(ar).toFixed(1)} -22 -38)`);
  $("#antL").setAttribute("transform", `rotate(${deg(al).toFixed(1)} 22 -38)`);
  $("#body").setAttribute("transform", `scale(${Math.max(0.55, Math.cos(body)).toFixed(3)} 1)`);
  $("#poseText").textContent =
    `pitch ${deg(pitch).toFixed(0)}° roll ${deg(roll).toFixed(0)}° yaw ${deg(yaw).toFixed(0)}° z ${(z * 1000).toFixed(0)}mm · ears ${deg(-ar).toFixed(0)}°/${deg(al).toFixed(0)}°`;
}

async function poll() {
  try {
    const s = await api(`/api/state?after=${after}`);
    document.querySelectorAll(".modes button[data-mode]").forEach((b) =>
      b.classList.toggle("on", b.dataset.mode === s.mode));
    for (const e of s.events) { onEvent(e); after = e.id; }
    drawPose(s.pose);
  } catch (_) { /* server restarting */ }
  setTimeout(poll, 80);
}
poll();
