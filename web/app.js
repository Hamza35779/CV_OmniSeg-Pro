/**
 * Image Segmentation — Real-World Problem Solver
 * Frontend application logic
 */

const API = "http://127.0.0.1:5000";

let currentDomain  = null;
let currentImg     = null;   // data URI or null (if using sample)
let usingSample    = false;

// ─────────────────────────────────────────────────────────────────────
// Domain configuration
// ─────────────────────────────────────────────────────────────────────

const DOMAINS = {
  medical: {
    icon: "🏥", label: "Medical Imaging",
    sub: "Lesion boundary & area measurement",
    accent: "#f87171",
    controls: `
      <div class="ctrl-row">
        <label class="ctrl-label">Scale (px / mm): <strong id="px-mm-val">3.0</strong></label>
        <input type="range" class="ctrl-slider" id="px-mm" min="1" max="10" step="0.5" value="3"
               oninput="document.getElementById('px-mm-val').textContent=parseFloat(this.value).toFixed(1)"/>
      </div>`,
    getBody: () => ({ px_per_mm: parseFloat(document.getElementById("px-mm")?.value || 3) }),
    renderResult: renderMedical,
  },
  traffic: {
    icon: "🚗", label: "Traffic & Roads",
    sub: "Lane detection & vehicle segmentation",
    accent: "#60a5fa",
    controls: "",
    getBody: () => ({}),
    renderResult: renderTraffic,
  },
  agriculture: {
    icon: "🌿", label: "Agriculture",
    sub: "Vegetation health & disease mapping",
    accent: "#4ade80",
    controls: "",
    getBody: () => ({}),
    renderResult: renderAgriculture,
  },
  industrial: {
    icon: "🏭", label: "Industrial QC",
    sub: "Surface defect detection & QC verdict",
    accent: "#fbbf24",
    controls: `
      <div class="ctrl-row">
        <label class="ctrl-label">Sensitivity: <strong id="sens-val">11</strong></label>
        <input type="range" class="ctrl-slider" id="sensitivity" min="5" max="31" step="2" value="11"
               oninput="document.getElementById('sens-val').textContent=this.value"/>
      </div>`,
    getBody: () => ({ sensitivity: parseInt(document.getElementById("sensitivity")?.value || 11) }),
    renderResult: renderIndustrial,
  },
  aerial: {
    icon: "🛰️", label: "Aerial / Satellite",
    sub: "Land-cover classification",
    accent: "#a78bfa",
    controls: `
      <div class="ctrl-row">
        <label class="ctrl-label">K-Means clusters: <strong id="k-aerial-val">4</strong></label>
        <input type="range" class="ctrl-slider" id="k-aerial" min="2" max="8" step="1" value="4"
               oninput="document.getElementById('k-aerial-val').textContent=this.value"/>
      </div>`,
    getBody: () => ({ k: parseInt(document.getElementById("k-aerial")?.value || 4) }),
    renderResult: renderAerial,
  },
  classical: {
    icon: "🔬", label: "Classical Methods",
    sub: "Thresholding · Contours · Watershed · K-Means · SLIC",
    accent: "#38bdf8",
    controls: `
      <div class="ctrl-row">
        <label class="ctrl-label">K-Means clusters: <strong id="k-val">4</strong></label>
        <input type="range" class="ctrl-slider" id="k-classic" min="2" max="8" step="1" value="4"
               oninput="document.getElementById('k-val').textContent=this.value"/>
      </div>
      <div class="ctrl-row">
        <label class="ctrl-label">Global threshold: <strong id="thresh-val">127</strong></label>
        <input type="range" class="ctrl-slider" id="thresh-classic" min="0" max="255" value="127"
               oninput="document.getElementById('thresh-val').textContent=this.value"/>
      </div>`,
    getBody: () => ({
      k: parseInt(document.getElementById("k-classic")?.value || 4),
      thresh_val: parseInt(document.getElementById("thresh-classic")?.value || 127),
    }),
    renderResult: renderClassical,
  },
};

// ─────────────────────────────────────────────────────────────────────
// Domain selection
// ─────────────────────────────────────────────────────────────────────

function selectDomain(name) {
  currentDomain = name;
  const cfg = DOMAINS[name];

  // Highlight card
  document.querySelectorAll(".domain-card").forEach(c => c.classList.remove("active"));
  document.getElementById(`card-${name}`).classList.add("active");

  // Nav label
  document.getElementById("nav-mode-label").textContent = cfg.label;

  // Show workspace
  document.getElementById("workspace").style.display = "grid";

  // Sidebar
  document.getElementById("sidebar-icon").textContent    = cfg.icon;
  document.getElementById("sidebar-title").textContent   = cfg.label;
  document.getElementById("sidebar-subtitle").textContent = cfg.sub;
  document.getElementById("domain-controls").innerHTML   = cfg.controls;

  // Reset results
  document.getElementById("results-title").textContent = `${cfg.icon} ${cfg.label} — Results`;
  document.getElementById("results-tabs").innerHTML = "";
  document.getElementById("results-grid").innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">${cfg.icon}</div>
      <div class="empty-text">Load an image and click Analyse</div>
    </div>`;
  document.getElementById("data-panel").innerHTML = "";
  document.getElementById("metrics-list").innerHTML = "";

  hide("summary-card");

  // Show load / analyse buttons
  document.getElementById("analyse-btn").style.display = "flex";
  document.getElementById("analyse-label").textContent = `Analyse — ${cfg.label}`;

  // Scroll to workspace
  document.getElementById("workspace").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ─────────────────────────────────────────────────────────────────────
// Image loading
// ─────────────────────────────────────────────────────────────────────

function onUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    currentImg   = e.target.result;
    usingSample  = false;
    showPreview(currentImg, `${file.name} · ${(file.size/1024).toFixed(1)} KB`);
  };
  reader.readAsDataURL(file);
}

async function loadDomainSample() {
  if (!currentDomain) return;
  setStatus("busy", "Loading…");
  try {
    const res  = await fetch(`${API}/domain/sample/${currentDomain}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    currentImg  = data.image;
    usingSample = true;
    showPreview(data.image, `Built-in sample: ${currentDomain}`);
    setStatus("ready", "Ready");
  } catch {
    setStatus("error", "Server unreachable");
    alert("Could not connect to the server. Make sure server.py is running.");
  }
}

function showPreview(uri, meta) {
  document.getElementById("preview-img").src  = uri;
  document.getElementById("preview-meta").textContent = meta;
  show("preview-wrap");
}

// ─────────────────────────────────────────────────────────────────────
// Run analysis
// ─────────────────────────────────────────────────────────────────────

async function runDomain() {
  if (!currentDomain || !currentImg) {
    alert("Please load an image first.");
    return;
  }
  const cfg = DOMAINS[currentDomain];
  const btn = document.getElementById("analyse-btn");
  btn.disabled = true;
  document.getElementById("analyse-icon").textContent  = "⏳";
  document.getElementById("analyse-label").textContent = "Analysing…";
  setStatus("busy", "Processing…");

  showGridLoading();

  try {
    const body = buildBody(cfg.getBody());
    let url, res, data;

    if (currentDomain === "classical") {
      url  = `${API}/segment/all`;
      res  = await fetch(url, post(body));
      data = await res.json();
    } else {
      url  = `${API}/domain/${currentDomain}`;
      res  = await fetch(url, post(body));
      data = await res.json();
    }

    if (data.error) throw new Error(data.error);

    cfg.renderResult(data);
    setStatus("ready", "Done");

  } catch (e) {
    setStatus("error", "Error");
    document.getElementById("results-grid").innerHTML =
      `<div class="empty-state"><div class="empty-icon">⚠️</div><div class="empty-text">${e.message}</div></div>`;
  } finally {
    btn.disabled = false;
    document.getElementById("analyse-icon").textContent  = "⚡";
    document.getElementById("analyse-label").textContent = `Analyse — ${cfg.label}`;
  }
}

function buildBody(extra = {}) {
  if (usingSample) return { domain_sample: currentDomain, ...extra };
  return { image: currentImg, ...extra };
}

// ─────────────────────────────────────────────────────────────────────
// Domain renderers
// ─────────────────────────────────────────────────────────────────────

function renderMedical(d) {
  renderGrid([
    { src: d.original,      label: "Original" },
    { src: d.binary_mask,   label: "Binary Mask" },
    { src: d.cleaned_mask,  label: "Cleaned Mask" },
    { src: d.annotated,     label: `Annotated (${d.lesion_count} lesions)` },
    { src: d.heatmap,       label: "Intensity Heatmap" },
  ]);

  setSummary(d.summary);

  setMetrics([
    { k: "Lesions Detected", v: d.lesion_count },
    { k: "Total Area (mm²)",  v: d.total_area_mm2 },
    { k: "Area (% of frame)", v: d.total_area_pct + "%" },
    { k: "Scale (px/mm)",     v: d.px_per_mm },
  ]);

  let tableRows = d.lesions.map(l => `
    <tr>
      <td>L${l.id}</td>
      <td>${l.area_mm2} mm²</td>
      <td>${l.perimeter_mm} mm</td>
      <td>${l.pct_of_image}%</td>
      <td><span class="verdict ${l.severity === 'Critical' ? 'fail' : l.severity === 'Severe' ? 'warn' : 'pass'}">${l.severity}</span></td>
    </tr>`).join("");

  setDataPanel(`
    <div class="panel-block">
      <div class="panel-block-title">Lesion Measurements</div>
      <table class="dt">
        <thead><tr><th>ID</th><th>Area</th><th>Perimeter</th><th>% Frame</th><th>Severity</th></tr></thead>
        <tbody>${tableRows || "<tr><td colspan='5' style='text-align:center;color:var(--t2)'>No lesions detected</td></tr>"}</tbody>
      </table>
    </div>`);
}

function renderTraffic(d) {
  renderGrid([
    { src: d.original,      label: "Original" },
    { src: d.lane_overlay,  label: `Lane Lines (${d.lane_count})` },
    { src: d.vehicle_mask,  label: `Vehicle Mask (${d.vehicle_count})` },
    { src: d.road_mask,     label: "Drivable Area" },
    { src: d.combined,      label: "Combined Output" },
  ]);

  setSummary(d.summary);

  setMetrics([
    { k: "Lane Lines",       v: d.lane_count },
    { k: "Vehicles",         v: d.vehicle_count },
    { k: "Near-Zone Danger", v: d.danger_count, cls: d.danger_count > 0 ? "fail" : "pass" },
  ]);

  const vRows = d.vehicles.map(v => `
    <tr>
      <td>${v.id}</td>
      <td>${v.zone === 'near' ? '<span class="verdict fail">Near</span>' : '<span class="verdict pass">Far</span>'}</td>
      <td>${v.area.toLocaleString()} px²</td>
      <td>(${v.bbox[0]}, ${v.bbox[1]})</td>
    </tr>`).join("");

  const lRows = d.lanes.map(l => `
    <tr><td>${l.id}</td><td>${l.side}</td><td>${l.angle}°</td></tr>`).join("");

  setDataPanel(`
    <div class="panel-block">
      <div class="panel-block-title">Detected Vehicles</div>
      <table class="dt">
        <thead><tr><th>#</th><th>Zone</th><th>Area</th><th>Position</th></tr></thead>
        <tbody>${vRows || "<tr><td colspan='4' style='text-align:center;color:var(--t2)'>None detected</td></tr>"}</tbody>
      </table>
    </div>
    <div class="panel-block">
      <div class="panel-block-title">Lane Lines</div>
      <table class="dt">
        <thead><tr><th>#</th><th>Side</th><th>Angle</th></tr></thead>
        <tbody>${lRows || "<tr><td colspan='3' style='text-align:center;color:var(--t2)'>None detected</td></tr>"}</tbody>
      </table>
    </div>`);
}

function renderAgriculture(d) {
  const s = d.stats;
  renderGrid([
    { src: d.original,     label: "Original" },
    { src: d.exg_map,      label: "Excess Green Index (ExG)" },
    { src: d.veg_mask,     label: "Vegetation Mask" },
    { src: d.disease_mask, label: "Disease Zones" },
    { src: d.healthy_mask, label: "Healthy Vegetation" },
    { src: d.annotated,    label: "Annotated Output" },
  ]);

  setSummary(d.summary);

  const scoreColor = s.crop_health_score >= 80 ? "pass" : (s.crop_health_score >= 50 ? "warn" : "fail");
  setMetrics([
    { k: "Health Score",     v: `${s.crop_health_score}/100`, cls: scoreColor },
    { k: "Vegetation Cover", v: `${s.veg_coverage_pct}%` },
    { k: "Disease Area",     v: `${s.disease_pct}%`, cls: s.disease_pct > 10 ? "fail" : "pass" },
    { k: "Disease in Veg",   v: `${s.disease_ratio_in_veg}%` },
  ]);

  const classes = [
    { name: "Healthy Crop", pct: s.healthy_pct,   color: "#4ade80" },
    { name: "Disease Zone", pct: s.disease_pct,   color: "#fbbf24" },
    { name: "Bare Soil",    pct: round(s.soil_px/(s.total_pixels)*100,2), color: "#a16207" },
  ];

  const bars = classes.map(c => `
    <div>
      <div style="display:flex;justify-content:space-between;margin-bottom:.3rem">
        <span style="font-size:12px;color:var(--t2)">${c.name}</span>
        <span style="font-size:12px;font-family:var(--mono);color:var(--t1)">${c.pct}%</span>
      </div>
      <div class="coverage-bar-track">
        <div class="coverage-bar-fill" style="width:${c.pct}%;background:${c.color}"></div>
      </div>
    </div>`).join("");

  setDataPanel(`
    <div class="panel-block">
      <div class="panel-block-title">Land Coverage Breakdown</div>
      <div style="display:flex;flex-direction:column;gap:.85rem">${bars}</div>
    </div>`);
}

function renderIndustrial(d) {
  renderGrid([
    { src: d.original,    label: "Original Surface" },
    { src: d.normalised,  label: "Normalised" },
    { src: d.defect_mask, label: "Defect Mask" },
    { src: d.annotated,   label: `Annotated (${d.defect_count} defects)` },
    { src: d.heatmap,     label: "Anomaly Heatmap" },
  ]);

  setSummary(d.summary);

  const vclass = d.verdict === "PASS" ? "pass" : d.verdict === "MARGINAL" ? "marginal" : "fail";
  setMetrics([
    { k: "QC Verdict",    v: d.verdict,          cls: vclass },
    { k: "Defects Found", v: d.defect_count,     cls: d.defect_count > 5 ? "fail" : "pass" },
    { k: "Defect Area %", v: d.defect_pct + "%", cls: d.defect_pct > 2 ? "fail" : "pass" },
  ]);

  const rows = d.defects.map(df => `
    <tr>
      <td>${df.id}</td>
      <td>${df.type}</td>
      <td>${df.area_px.toLocaleString()}</td>
      <td>${df.circularity}</td>
      <td><span class="verdict ${df.severity === 'Major' ? 'fail' : df.severity === 'Moderate' ? 'warn' : 'pass'}">${df.severity}</span></td>
    </tr>`).join("");

  setDataPanel(`
    <div class="panel-block">
      <div class="panel-block-title">
        QC Verdict: <span class="verdict ${vclass}">${d.verdict}</span>
        &nbsp;·&nbsp; Defect Surface: ${d.defect_pct}%
      </div>
      <table class="dt">
        <thead><tr><th>#</th><th>Type</th><th>Area (px²)</th><th>Circularity</th><th>Severity</th></tr></thead>
        <tbody>${rows || "<tr><td colspan='5' style='text-align:center;color:var(--t2)'>No defects detected</td></tr>"}</tbody>
      </table>
    </div>`);
}

function renderAerial(d) {
  renderGrid([
    { src: d.original,   label: "Original" },
    { src: d.class_map,  label: "HSV Class Map" },
    { src: d.kmeans_map, label: "K-Means Map" },
    { src: d.overlay,    label: "Overlay (50/50)" },
  ]);

  setSummary(d.summary);

  const colors = { Water: "#3b82f6", Vegetation: "#22c55e", "Urban/Built": "#94a3b8", "Bare Soil": "#ca8a04" };
  setMetrics(d.classes.map(c => ({ k: c.name, v: `${c.pct}%` })));

  const bars = d.classes.map(c => `
    <div>
      <div style="display:flex;justify-content:space-between;margin-bottom:.3rem">
        <span style="font-size:12px;color:var(--t2)">${c.name}</span>
        <span style="font-size:12px;font-family:var(--mono);color:var(--t1)">${c.pct}%</span>
      </div>
      <div class="coverage-bar-track">
        <div class="coverage-bar-fill" style="width:${c.pct}%;background:${colors[c.name] || '#6b7280'}"></div>
      </div>
    </div>`).join("");

  setDataPanel(`
    <div class="panel-block">
      <div class="panel-block-title">Land Cover Distribution (Dominant: ${d.dominant})</div>
      <div style="display:flex;flex-direction:column;gap:.85rem">${bars}</div>
    </div>`);
}

function renderClassical(d) {
  // Multi-tab view
  const tabs = [
    { id: "ct-threshold",  label: "Thresholding" },
    { id: "ct-contours",   label: "Contours" },
    { id: "ct-watershed",  label: "Watershed" },
    { id: "ct-kmeans",     label: "K-Means" },
    { id: "ct-slic",       label: "Superpixels" },
  ];

  const tabHtml = tabs.map((t, i) => `
    <button class="rtab ${i===0?"active":""}" id="btn-${t.id}"
            onclick="switchClassicalTab('${t.id}')">${t.label}</button>`
  ).join("");

  document.getElementById("results-tabs").innerHTML = tabHtml;

  const panels = {
    "ct-threshold": [
      { src: d.threshold.original,          label: "Original" },
      { src: d.threshold.global?.binary,    label: "Global Binary" },
      { src: d.threshold.adaptive?.binary,  label: "Adaptive Binary" },
      { src: d.threshold.otsu?.binary,      label: `Otsu (T=${d.threshold.otsu?.otsu_threshold})` },
    ],
    "ct-contours": [
      { src: d.contours.original,  label: "Original" },
      { src: d.contours.mask,      label: "Mask" },
      { src: d.contours.annotated, label: `Annotated (${d.contours.count} objects)` },
      { src: d.canny.edges,        label: "Canny Edges" },
      { src: d.canny.overlay,      label: "Edge Overlay" },
    ],
    "ct-watershed": [
      { src: d.watershed.original,         label: "Original" },
      { src: d.watershed.dist_transform,   label: "Distance Transform" },
      { src: d.watershed.result,           label: `Segmented (${d.watershed.region_count} regions)` },
      { src: d.watershed.boundary_overlay, label: "Boundaries" },
      { src: d.grabcut.fg_mask,            label: "GrabCut Mask" },
      { src: d.grabcut.result,             label: "GrabCut Result" },
    ],
    "ct-kmeans": [
      { src: d.kmeans.original,   label: "Original" },
      { src: d.kmeans.segmented,  label: `K-Means (k=${d.kmeans.k})` },
      { src: d.kmeans.overlay,    label: "Overlay" },
    ],
    "ct-slic": [
      { src: d.superpixels.original,   label: "Original" },
      { src: d.superpixels.boundaries, label: `SLIC Boundaries (${d.superpixels.n_superpixels})` },
      { src: d.superpixels.mean_color, label: "Mean-Color Image" },
    ],
  };

  window._classicalPanels = panels;
  renderClassicalTab("ct-threshold");

  // K-Means palette
  if (d.kmeans?.clusters) {
    const chips = d.kmeans.clusters.map(c => `
      <div class="palette-chip">
        <div class="palette-chip-color" style="background:${c.color_hex}"></div>
        <div>${c.pct}%</div>
        <div>${c.color_hex}</div>
      </div>`).join("");
    setDataPanel(`
      <div class="panel-block">
        <div class="panel-block-title">K-Means Color Palette</div>
        <div class="palette-row">${chips}</div>
      </div>`);
  }
}

function switchClassicalTab(id) {
  document.querySelectorAll(".rtab").forEach(b => b.classList.remove("active"));
  document.getElementById(`btn-${id}`).classList.add("active");
  renderClassicalTab(id);
}

function renderClassicalTab(id) {
  if (!window._classicalPanels) return;
  renderGrid(window._classicalPanels[id] || []);
}

// ─────────────────────────────────────────────────────────────────────
// UI helpers
// ─────────────────────────────────────────────────────────────────────

function renderGrid(images) {
  const grid = document.getElementById("results-grid");
  if (!images || !images.length) {
    grid.innerHTML = '<div class="empty-state"><div class="empty-text">No images to display</div></div>';
    return;
  }
  grid.innerHTML = images
    .filter(im => im && im.src)
    .map((im, i) => `
      <div class="img-card" data-idx="${i}">
        <img src="${im.src}" alt="${im.label}" loading="lazy"/>
        <div class="img-label">${im.label}</div>
      </div>`).join("");

  grid.querySelectorAll(".img-card").forEach((card, i) => {
    card.addEventListener("click", () => openModal(images[i].src, images[i].label));
  });
}

function showGridLoading() {
  document.getElementById("results-grid").innerHTML =
    Array(4).fill('<div class="img-card loading"><div class="spinner"></div></div>').join("");
}

function setSummary(text) {
  document.getElementById("summary-text").textContent = text || "";
  show("summary-card");
}

function setMetrics(rows) {
  document.getElementById("metrics-list").innerHTML = rows.map(r => `
    <div class="metric-row">
      <span class="metric-key">${r.k}</span>
      <span class="metric-val ${r.cls || ''}">${r.v}</span>
    </div>`).join("");
}

function setDataPanel(html) {
  document.getElementById("data-panel").innerHTML = html;
}

function openModal(src, cap) {
  document.getElementById("modal-img").src = src;
  document.getElementById("modal-cap").textContent = cap;
  document.getElementById("modal-bg").classList.add("open");
}

function closeModal() {
  document.getElementById("modal-bg").classList.remove("open");
}

function show(id) { document.getElementById(id).style.display = "" }
function hide(id) { document.getElementById(id).style.display = "none" }

function setStatus(state, text) {
  document.getElementById("status-dot").className   = `status-dot ${state}`;
  document.getElementById("status-text").textContent = text;
}

function post(body) {
  return {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  };
}

function round(v, dp) { return Math.round(v * 10**dp) / 10**dp; }

// ─────────────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────────────

document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });
document.addEventListener("DOMContentLoaded", () => setStatus("ready", "Ready"));
