/** Platform management: models, datasets, devices, alerts, training jobs */

const platformState = {
  models: [],
  datasets: [],
  devices: [],
  alerts: [],
  trainingJobs: [],
  analysisStatus: [],
};

const platformPageState = {
  model: { page: 1, pageSize: 10 },
  dataset: { page: 1, pageSize: 10 },
  device: { page: 1, pageSize: 10 },
  alert: { page: 1, pageSize: 12 },
  trainJob: { page: 1, pageSize: 10 },
};

function platformEsc(s) {
  if (s == null) return "";
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function formatUploadTime(iso) {
  return typeof formatDateTime === "function" ? formatDateTime(iso) : (iso || "—");
}

function getFilteredDevices() {
  const auditOpts = getAuditFilterOpts("device");
  return platformState.devices.filter((d) =>
    matchAuditFilters(d, {
      keyword: auditOpts.keyword,
      createdBy: auditOpts.createdBy,
      createdFrom: auditOpts.createdFrom,
      createdTo: auditOpts.createdTo,
      extraFields: [d.name, d.id, d.source, d.device_type, d.device_type === "video" ? "视频流" : "图片流"],
    })
  );
}

function getFilteredTrainingJobs() {
  const auditOpts = getAuditFilterOpts("trainJob");
  const state = document.getElementById("trainJobFilterState")?.value || "";
  return platformState.trainingJobs.filter((j) => {
    if (state && j.state !== state) return false;
    return matchAuditFilters(j, {
      keyword: auditOpts.keyword,
      createdBy: auditOpts.createdBy,
      createdFrom: auditOpts.createdFrom,
      createdTo: auditOpts.createdTo,
      extraFields: [j.name, j.id, j.model_name, j.dataset_name, j.state, j.message],
    });
  });
}

async function loadPlatformModels() {
  const data = await api("/api/platform/models");
  platformState.models = data.items || [];
  renderModelTable();
  refreshModelSelects();
}

async function loadPlatformDatasets() {
  const data = await api("/api/platform/datasets");
  platformState.datasets = data.items || [];
  renderDatasetTable();
  refreshDatasetSelects();
}

async function loadPlatformDevices() {
  const data = await api("/api/platform/devices");
  platformState.devices = data.items || [];
  populateAuditUserSelect(document.getElementById("deviceFilterCreator"), platformState.devices);
  renderDeviceTable();
  refreshDeviceSelects();
}

async function loadPlatformAlerts(page) {
  if (page) platformPageState.alert.page = page;
  const p = platformPageState.alert;
  const data = await api(`/api/platform/alerts?page=${p.page}&page_size=${p.pageSize}`);
  platformState.alerts = data.items || [];
  renderAlertGrid(data);
}

async function loadTrainingJobs() {
  const data = await api("/api/platform/training-jobs");
  platformState.trainingJobs = data.items || [];
  populateAuditUserSelect(document.getElementById("trainJobFilterCreator"), platformState.trainingJobs);
  renderTrainingJobTable();
}

async function loadAnalysisStatus() {
  try {
    const data = await api("/api/platform/analysis/status");
    platformState.analysisStatus = data.tasks || [];
  } catch (_) {
    platformState.analysisStatus = [];
  }
}

function refreshModelSelects() {
  const opts = platformState.models.map((m) =>
    `<option value="${m.id}">${platformEsc(m.name)} [${m.model_type}] ${m.in_use ? "●" : ""}</option>`
  ).join("");
  ["modalInstModel", "modalTrainModelSelect"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    const prev = el.value;
    el.innerHTML = `<option value="">— 选择模型 —</option>${opts}`;
    if (prev) el.value = prev;
  });
}

function refreshDatasetSelects() {
  const opts = platformState.datasets.map((d) =>
    `<option value="${d.id}">${platformEsc(d.name)} (train:${d.train_count})</option>`
  ).join("");
  ["modalTrainDatasetSelect"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    const prev = el.value;
    el.innerHTML = `<option value="">— 选择数据集 —</option>${opts}`;
    if (prev) el.value = prev;
  });
}

function refreshDeviceSelects() {
  const filter = document.getElementById("alertDeviceFilter");
  if (filter) {
    filter.innerHTML = `<option value="">全部设备</option>${platformState.devices.map((d) =>
      `<option value="${d.id}">${platformEsc(d.name)}</option>`).join("")}`;
  }
}

function renderModelTable() {
  const keyword = document.getElementById("modelSearch")?.value.trim() || "";
  let rows = platformState.models.filter((m) =>
    !keyword || [m.name, m.id, m.model_type, m.file_path, m.uploaded_by].join(" ").toLowerCase().includes(keyword.toLowerCase())
  );
  const pag = paginateRows(rows, platformPageState.model);
  const tbody = document.getElementById("modelTableBody");
  const empty = document.getElementById("modelTableEmpty");
  const countEl = document.getElementById("modelTableCount");
  if (countEl) countEl.textContent = `共 ${pag.total} 条`;
  if (!tbody) return;
  if (pag.total === 0) {
    tbody.innerHTML = "";
    if (empty) empty.hidden = false;
    renderTablePagination("#modelPagination", pag, "model", renderModelTable);
    return;
  }
  if (empty) empty.hidden = true;
  tbody.innerHTML = pag.slice.map((m) => `<tr>
    <td><strong>${platformEsc(m.name)}</strong></td>
    <td><span class="tag tag-${m.model_type === "yolo" ? "warn" : "info"}">${m.model_type}</span></td>
    <td class="cell-ellipsis" title="${platformEsc(m.file_path)}">${platformEsc(m.file_path.split("/").pop())}</td>
    <td>${platformEsc(m.version)}</td>
    <td>${m.source === "deploy" ? "训练部署" : "上传"}</td>
    <td class="cell-muted">${formatUploadTime(m.created_at)}</td>
    <td>${platformEsc(m.uploaded_by) || "—"}</td>
    <td>${m.in_use ? '<span class="tag">使用中</span>' : "—"}</td>
    <td class="col-actions">
      <div class="action-group">
        <button class="btn btn-secondary btn-sm" data-model-action="lineage" data-id="${m.id}">来源</button>
        <button class="btn btn-danger btn-sm" data-model-action="delete" data-id="${m.id}" ${m.in_use ? "disabled" : ""}>删除</button>
      </div>
    </td>
  </tr>`).join("");
  tbody.querySelectorAll("[data-model-action]").forEach((btn) => {
    btn.onclick = async () => {
      const id = btn.dataset.id;
      if (btn.dataset.modelAction === "delete") {
        if (!confirm("确定删除该模型？")) return;
        try {
          await api(`/api/platform/models/${id}`, { method: "DELETE" });
          toast("模型已删除");
          loadPlatformModels();
        } catch (e) { toast(e.message, true); }
      } else if (btn.dataset.modelAction === "lineage") {
        try {
          const data = await api(`/api/platform/models/${id}/lineage`);
          const html = (data.lineage || []).map((l) =>
            `<div class="lineage-item" style="padding:6px 0;border-bottom:1px solid var(--border)">
              ${"→".repeat(l.depth)} ${platformEsc(l.name)} <span class="cell-muted">[${l.model_type} ${l.version} · ${l.source}]</span>
            </div>`).join("") || "<p class=\"hint\">无来源记录</p>";
          openModal({ title: "模型来源链", bodyHtml: html, confirmText: "关闭", onConfirm: closeModal });
          document.getElementById("modalCancel").hidden = true;
          setTimeout(() => { document.getElementById("modalCancel").hidden = false; }, 100);
        } catch (e) { toast(e.message, true); }
      }
    };
  });
  renderTablePagination("#modelPagination", pag, "model", renderModelTable);
}

function renderDatasetTable() {
  const keyword = document.getElementById("datasetSearch")?.value.trim() || "";
  let rows = platformState.datasets.filter((d) =>
    !keyword || [d.name, d.id, d.path, d.uploaded_by].join(" ").toLowerCase().includes(keyword.toLowerCase())
  );
  const pag = paginateRows(rows, platformPageState.dataset);
  const tbody = document.getElementById("datasetTableBody");
  const empty = document.getElementById("datasetTableEmpty");
  const countEl = document.getElementById("datasetTableCount");
  if (countEl) countEl.textContent = `共 ${pag.total} 条`;
  if (!tbody) return;
  if (pag.total === 0) {
    tbody.innerHTML = "";
    if (empty) empty.hidden = false;
    renderTablePagination("#datasetPagination", pag, "dataset", renderDatasetTable);
    return;
  }
  if (empty) empty.hidden = true;
  tbody.innerHTML = pag.slice.map((d) => `<tr>
    <td><strong>${platformEsc(d.name)}</strong></td>
    <td class="cell-muted">${platformEsc(d.id)}</td>
    <td>${d.train_count}</td>
    <td>${d.valid_count}</td>
    <td>${d.test_count}</td>
    <td class="cell-muted">${formatUploadTime(d.created_at)}</td>
    <td>${platformEsc(d.uploaded_by) || "—"}</td>
    <td>${(d.class_names || []).map(platformEsc).join(", ") || "—"}</td>
    <td class="col-actions">
      <div class="action-group">
        <button class="btn btn-primary btn-sm" data-ds-action="browse" data-id="${d.id}">浏览</button>
        <button class="btn btn-danger btn-sm" data-ds-action="delete" data-id="${d.id}">删除</button>
      </div>
    </td>
  </tr>`).join("");
  tbody.querySelectorAll("[data-ds-action]").forEach((btn) => {
    btn.onclick = async () => {
      const id = btn.dataset.id;
      if (btn.dataset.dsAction === "browse") {
        const ds = platformState.datasets.find((d) => d.id === id);
        openDatasetViewer(id, ds?.name || id);
      } else {
        if (!confirm("确定删除该数据集？")) return;
        try {
          await api(`/api/platform/datasets/${id}`, { method: "DELETE" });
          toast("数据集已删除");
          loadPlatformDatasets();
        } catch (e) { toast(e.message, true); }
      }
    };
  });
  renderTablePagination("#datasetPagination", pag, "dataset", renderDatasetTable);
}

const datasetViewerState = {
  datasetId: null,
  datasetName: "",
  split: "train",
  images: [],
  index: 0,
  showLabels: true,
  annotations: [],
  classNames: [],
  loading: false,
};

const DV_LABEL_COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4"];

function datasetPreviewUrl(datasetId, itemPath) {
  if (!datasetId || !itemPath) return "";
  const base = `/api/platform/datasets/${datasetId}/file?path=${encodeURIComponent(itemPath)}`;
  return authUrl(base);
}

async function openDatasetViewer(datasetId, datasetName = "") {
  datasetViewerState.datasetId = datasetId;
  datasetViewerState.datasetName = datasetName;
  datasetViewerState.split = "train";
  datasetViewerState.index = 0;
  datasetViewerState.showLabels = true;
  const viewer = document.getElementById("datasetViewer");
  const splitSel = document.getElementById("dvSplitSelect");
  const showLabelsEl = document.getElementById("dvShowLabels");
  if (splitSel) splitSel.value = "train";
  if (showLabelsEl) showLabelsEl.checked = true;
  if (viewer) viewer.hidden = false;
  document.body.style.overflow = "hidden";
  await loadDatasetViewerImages();
}

function closeDatasetViewer() {
  const viewer = document.getElementById("datasetViewer");
  if (viewer) viewer.hidden = true;
  document.body.style.overflow = "";
  datasetViewerState.images = [];
  datasetViewerState.annotations = [];
}

async function loadDatasetViewerImages() {
  const { datasetId, split } = datasetViewerState;
  if (!datasetId) return;
  datasetViewerState.loading = true;
  const meta = document.getElementById("dvMeta");
  if (meta) meta.textContent = "加载中...";
  try {
    const data = await api(`/api/platform/datasets/${datasetId}/images?split=${split}`);
    datasetViewerState.images = data.items || [];
    datasetViewerState.datasetName = data.dataset_name || datasetViewerState.datasetName;
    datasetViewerState.classNames = data.class_names || [];
    datasetViewerState.index = Math.min(datasetViewerState.index, Math.max(0, datasetViewerState.images.length - 1));
    const nameEl = document.getElementById("dvDatasetName");
    if (nameEl) nameEl.textContent = datasetViewerState.datasetName || "数据集浏览";
    if (meta) {
      meta.textContent = `${split} · 共 ${data.total} 张 · 已标注 ${data.labeled_count || 0} 张`;
    }
    if (datasetViewerState.images.length === 0) {
      document.getElementById("dvImage").removeAttribute("src");
      document.getElementById("dvImageName").textContent = "该划分下暂无图片";
      updateDatasetViewerNav();
      return;
    }
    await showDatasetViewerImage(datasetViewerState.index);
  } catch (e) {
    toast(e.message, true);
    closeDatasetViewer();
  } finally {
    datasetViewerState.loading = false;
  }
}

async function showDatasetViewerImage(index) {
  const images = datasetViewerState.images;
  if (!images.length || index < 0 || index >= images.length) return;
  datasetViewerState.index = index;
  const item = images[index];
  const imgEl = document.getElementById("dvImage");
  const nameEl = document.getElementById("dvImageName");
  const toggleWrap = document.getElementById("dvLabelToggleWrap");
  const labelPanel = document.getElementById("dvLabelPanel");
  const labelList = document.getElementById("dvLabelList");
  const src = datasetPreviewUrl(datasetViewerState.datasetId, item.path);

  updateDatasetViewerNav();

  if (nameEl) {
    nameEl.textContent = `${index + 1} / ${images.length} · ${item.name}${item.has_labels ? " · 已标注" : ""}`;
  }

  datasetViewerState.annotations = [];
  if (item.has_labels) {
    try {
      const labelData = await api(
        `/api/platform/datasets/${datasetViewerState.datasetId}/labels?path=${encodeURIComponent(item.path)}`
      );
      datasetViewerState.annotations = labelData.annotations || [];
    } catch (_) {
      datasetViewerState.annotations = [];
    }
  }

  const hasAnns = datasetViewerState.annotations.length > 0;
  if (toggleWrap) toggleWrap.hidden = !hasAnns;
  if (!hasAnns) {
    if (labelPanel) labelPanel.hidden = true;
    if (labelList) labelList.innerHTML = '<p class="hint">当前图片无标注文件</p>';
  }

  if (imgEl) {
    imgEl.onload = () => {
      renderDatasetViewerOverlay();
      renderDatasetViewerLabelList();
    };
    imgEl.src = src;
  }
}

function updateDatasetViewerNav() {
  const { images, index } = datasetViewerState;
  const prev = document.getElementById("dvPrev");
  const next = document.getElementById("dvNext");
  if (prev) prev.disabled = index <= 0;
  if (next) next.disabled = !images.length || index >= images.length - 1;
}

function renderDatasetViewerOverlay() {
  const canvas = document.getElementById("dvCanvas");
  const img = document.getElementById("dvImage");
  if (!canvas || !img || !img.complete || !img.naturalWidth) return;

  const show = datasetViewerState.showLabels && datasetViewerState.annotations.length > 0;
  const wrap = img.parentElement;
  const displayW = img.clientWidth;
  const displayH = img.clientHeight;
  if (!displayW || !displayH) return;

  canvas.width = displayW;
  canvas.height = displayH;
  canvas.style.width = `${displayW}px`;
  canvas.style.height = `${displayH}px`;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, displayW, displayH);
  if (!show) return;

  const sx = displayW / img.naturalWidth;
  const sy = displayH / img.naturalHeight;

  datasetViewerState.annotations.forEach((ann) => {
    const cx = ann.cx * displayW;
    const cy = ann.cy * displayH;
    const bw = ann.w * displayW;
    const bh = ann.h * displayH;
    const x = cx - bw / 2;
    const y = cy - bh / 2;
    const color = DV_LABEL_COLORS[ann.class_id % DV_LABEL_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, bw, bh);
    ctx.fillStyle = color;
    ctx.font = "12px sans-serif";
    const label = `${ann.class_name}`;
    const tw = ctx.measureText(label).width + 8;
    ctx.fillRect(x, Math.max(0, y - 18), tw, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + 4, Math.max(12, y - 5));
  });
}

function renderDatasetViewerLabelList() {
  const panel = document.getElementById("dvLabelPanel");
  const list = document.getElementById("dvLabelList");
  const show = datasetViewerState.showLabels && datasetViewerState.annotations.length > 0;
  if (!list) return;
  if (!show) {
    if (panel) panel.hidden = true;
    return;
  }
  if (panel) panel.hidden = false;
  list.innerHTML = datasetViewerState.annotations.map((ann) => {
    const x1 = ((ann.cx - ann.w / 2) * 100).toFixed(1);
    const y1 = ((ann.cy - ann.h / 2) * 100).toFixed(1);
    const x2 = ((ann.cx + ann.w / 2) * 100).toFixed(1);
    const y2 = ((ann.cy + ann.h / 2) * 100).toFixed(1);
    return `<div class="det-item">
      <span><strong>${platformEsc(ann.class_name)}</strong> #${ann.class_id}</span>
      <span>YOLO: cx=${ann.cx.toFixed(4)} cy=${ann.cy.toFixed(4)} w=${ann.w.toFixed(4)} h=${ann.h.toFixed(4)}</span>
      <span style="grid-column:1/-1;color:var(--muted)">bbox %(≈): [${x1}, ${y1}, ${x2}, ${y2}]</span>
    </div>`;
  }).join("");
}

function datasetViewerStep(delta) {
  if (datasetViewerState.loading) return;
  const next = datasetViewerState.index + delta;
  if (next < 0 || next >= datasetViewerState.images.length) return;
  showDatasetViewerImage(next);
}

function initDatasetViewer() {
  document.getElementById("dvClose")?.addEventListener("click", closeDatasetViewer);
  document.getElementById("dvPrev")?.addEventListener("click", () => datasetViewerStep(-1));
  document.getElementById("dvNext")?.addEventListener("click", () => datasetViewerStep(1));
  document.getElementById("dvSplitSelect")?.addEventListener("change", async (e) => {
    datasetViewerState.split = e.target.value;
    datasetViewerState.index = 0;
    await loadDatasetViewerImages();
  });
  document.getElementById("dvShowLabels")?.addEventListener("change", (e) => {
    datasetViewerState.showLabels = e.target.checked;
    renderDatasetViewerOverlay();
    renderDatasetViewerLabelList();
  });
  window.addEventListener("resize", () => {
    if (!document.getElementById("datasetViewer")?.hidden) renderDatasetViewerOverlay();
  });
  document.addEventListener("keydown", (e) => {
    const viewer = document.getElementById("datasetViewer");
    if (viewer?.hidden) return;
    if (e.key === "Escape") closeDatasetViewer();
    else if (e.key === "ArrowLeft") { e.preventDefault(); datasetViewerStep(-1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); datasetViewerStep(1); }
  });
}

function renderDeviceTable() {
  const rows = getFilteredDevices();
  const pag = paginateRows(rows, platformPageState.device);
  const tbody = document.getElementById("deviceTableBody");
  const empty = document.getElementById("deviceTableEmpty");
  const countEl = document.getElementById("deviceTableCount");
  if (countEl) countEl.textContent = `共 ${pag.total} 条`;
  if (!tbody) return;
  if (pag.total === 0) {
    tbody.innerHTML = "";
    if (empty) empty.hidden = false;
    renderTablePagination("#devicePagination", pag, "device", renderDeviceTable);
    return;
  }
  if (empty) empty.hidden = true;
  tbody.innerHTML = pag.slice.map((d) => `<tr>
    <td><strong>${platformEsc(d.name)}</strong></td>
    <td><span class="tag">${d.device_type === "video" ? "视频流" : "图片流"}</span></td>
    <td class="cell-ellipsis" title="${platformEsc(d.source)}">${platformEsc(d.source)}</td>
    <td>${d.device_type === "image" ? d.poll_interval + "s" : "—"}</td>
    <td>${(d.roi || []).length} 区域</td>
    <td>${d.analysis_running ? '<span class="badge badge-running">分析中</span>' : (d.enabled ? "启用" : "禁用")}</td>
    ${renderAuditCells(d, platformEsc)}
    <td class="col-actions">
      <div class="action-group">
        <button class="btn btn-secondary btn-sm" data-dev-action="roi" data-id="${d.id}">配置 ROI</button>
        <button class="btn btn-secondary btn-sm" data-dev-action="edit" data-id="${d.id}">编辑</button>
        <button class="btn btn-danger btn-sm" data-dev-action="delete" data-id="${d.id}" ${d.analysis_running ? "disabled" : ""}>删除</button>
      </div>
    </td>
  </tr>`).join("");
  tbody.querySelectorAll("[data-dev-action]").forEach((btn) => {
    btn.onclick = async () => {
      const id = btn.dataset.id;
      if (btn.dataset.devAction === "edit") openDeviceModal(id);
      else if (btn.dataset.devAction === "roi") openRoiEditor(id);
      else {
        if (!confirm("确定删除该设备？")) return;
        try {
          await api(`/api/platform/devices/${id}`, { method: "DELETE" });
          toast("设备已删除");
          loadPlatformDevices();
        } catch (e) { toast(e.message, true); }
      }
    };
  });
  renderTablePagination("#devicePagination", pag, "device", renderDeviceTable);
}

function renderAlertGrid(data) {
  const grid = document.getElementById("alertGrid");
  if (!grid) return;
  const items = data.items || [];
  grid.innerHTML = items.length
    ? items.map((a) => `<div class="alert-card">
        <div class="alert-card-img" onclick="openLightbox('${a.image_url}')">
          <img src="${a.image_url}" alt="报警" loading="lazy" />
        </div>
        <div class="alert-card-body">
          <strong>${platformEsc(a.device_name)}</strong>
          <span class="cell-muted">${formatDateTime(a.alert_at)}</span>
          <span>实例: ${platformEsc(a.instance_name)}</span>
          <span class="tag tag-warn">置信度 ${(a.max_confidence * 100).toFixed(1)}%</span>
          <span>${a.detections?.length || 0} 个目标</span>
          <div class="alert-card-actions">
            <button type="button" class="btn btn-danger btn-sm" data-alert-del="${platformEsc(a.id)}">删除</button>
          </div>
        </div>
      </div>`).join("")
    : '<p class="hint">暂无报警记录</p>';
  grid.querySelectorAll("[data-alert-del]").forEach((btn) => {
    btn.onclick = async (e) => {
      e.stopPropagation();
      const id = btn.dataset.alertDel;
      if (!confirm("确定删除该报警记录？")) return;
      try {
        await api(`/api/platform/alerts/${id}`, { method: "DELETE" });
        toast("报警记录已删除");
        loadPlatformAlerts();
      } catch (err) { toast(err.message, true); }
    };
  });
  const pag = { total: data.total, page: data.page, pageSize: data.page_size, totalPages: Math.max(1, Math.ceil(data.total / data.page_size)), rangeStart: data.total ? (data.page - 1) * data.page_size + 1 : 0, rangeEnd: Math.min(data.page * data.page_size, data.total) };
  platformPageState.alert.page = data.page;
  renderTablePagination("#alertPagination", pag, "alert", () => loadPlatformAlerts());
}

function renderTrainingJobTable() {
  const rows = getFilteredTrainingJobs();
  const pag = paginateRows(rows, platformPageState.trainJob);
  const tbody = document.getElementById("trainJobTableBody");
  const empty = document.getElementById("trainJobTableEmpty");
  const countEl = document.getElementById("trainJobTableCount");
  if (countEl) countEl.textContent = `共 ${pag.total} 条`;
  if (!tbody) return;
  if (pag.total === 0) {
    tbody.innerHTML = "";
    if (empty) empty.hidden = false;
    renderTablePagination("#trainJobPagination", pag, "trainJob", renderTrainingJobTable);
    return;
  }
  if (empty) empty.hidden = true;
  const stateBadge = (s) => ({ completed: "ready", running: "running", failed: "error", pending: "stopped" }[s] || "stopped");
  tbody.innerHTML = pag.slice.map((j) => `<tr>
    <td><strong>${platformEsc(j.name)}</strong></td>
    <td>${platformEsc(j.model_name)}</td>
    <td>${platformEsc(j.dataset_name)}</td>
    <td><span class="badge badge-${stateBadge(j.state)}">${j.state}</span></td>
    <td>${j.epochs}</td>
    <td class="cell-muted">${platformEsc(j.message?.slice(0, 40))}</td>
    ${renderAuditCells(j, platformEsc)}
    <td class="col-actions">
      <div class="action-group">
        ${j.state === "pending" || j.state === "failed" ? `<button class="btn btn-primary btn-sm" data-job-action="start" data-id="${j.id}">训练</button>` : ""}
        ${j.state === "running" ? `<button class="btn btn-danger btn-sm" data-job-action="stop" data-id="${j.id}">停止</button>` : ""}
        <button class="btn btn-secondary btn-sm" data-job-action="log" data-id="${j.id}">日志</button>
        ${j.state === "completed" && !j.deployed_model_id ? `<button class="btn btn-primary btn-sm" data-job-action="deploy" data-id="${j.id}">部署</button>` : ""}
        ${j.deployed_model_id ? '<span class="tag">已部署</span>' : ""}
      </div>
    </td>
  </tr>`).join("");
  tbody.querySelectorAll("[data-job-action]").forEach((btn) => {
    btn.onclick = async () => {
      const id = btn.dataset.id;
      try {
        if (btn.dataset.jobAction === "start") {
          const r = await api("/api/train/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ job_id: id }) });
          toast(r.message);
          refreshStatus();
          loadTrainingJobs();
        } else if (btn.dataset.jobAction === "stop") {
          const r = await api(`/api/platform/training-jobs/${id}/stop`, { method: "POST" });
          toast(r.message);
          refreshStatus();
          loadTrainingJobs();
        } else if (btn.dataset.jobAction === "log") {
          openTrainJobLogModal(id);
        } else {
          const r = await api(`/api/platform/training-jobs/${id}/deploy`, { method: "POST" });
          toast(`已部署为模型: ${r.name}`);
          loadPlatformModels();
          loadTrainingJobs();
        }
      } catch (e) { toast(e.message, true); }
    };
  });
  renderTablePagination("#trainJobPagination", pag, "trainJob", renderTrainingJobTable);
}

function openUploadModelModal() {
  openModal({
    title: "上传模型",
    wide: true,
    confirmText: "上传",
    bodyHtml: `
      <div class="form-grid">
        <div><label>模型名称</label><input type="text" id="uploadModelName" placeholder="烟雾检测-v1" /></div>
        <div><label>模型类型</label><select id="uploadModelType"><option value="rf-detr">RF-DETR</option><option value="yolo">YOLO</option></select></div>
        <div class="full"><label>类别名（逗号分隔）</label><input type="text" id="uploadModelClasses" placeholder="Fire,Smoke" /></div>
        <div class="full"><label>权重文件 (.pth / .pt)</label><input type="file" id="uploadModelFile" accept=".pth,.pt,.bin,.onnx" /></div>
      </div>`,
    onConfirm: async () => {
      const file = document.getElementById("uploadModelFile").files[0];
      if (!file) throw new Error("请选择模型文件");
      setModalBusy(true, "上传中...");
      toast("正在上传模型，请勿重复点击");
      try {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("name", document.getElementById("uploadModelName").value.trim());
        fd.append("model_type", document.getElementById("uploadModelType").value);
        fd.append("class_names", document.getElementById("uploadModelClasses").value);
        const res = await fetch("/api/platform/models/upload", { method: "POST", headers: authHeaders(), body: fd });
        if (res.status === 401) {
          setModalBusy(false);
          redirectLogin();
          return;
        }
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "上传失败");
        modalBusy = false;
        toast("模型上传成功");
        closeModal();
        loadPlatformModels();
      } catch (e) {
        setModalBusy(false);
        throw e;
      }
    },
  });
}

function openUploadDatasetModal() {
  openModal({
    title: "上传数据集 (ZIP)",
    confirmText: "上传",
    bodyHtml: `
      <p class="hint">ZIP 内需包含 train、valid、test 文件夹及 data.yaml</p>
      <div class="form-grid">
        <div class="full"><label>数据集名称</label><input type="text" id="uploadDatasetName" placeholder="smoke_fire" /></div>
        <div class="full"><label>ZIP 文件</label><input type="file" id="uploadDatasetFile" accept=".zip" /></div>
      </div>`,
    onConfirm: async () => {
      const file = document.getElementById("uploadDatasetFile").files[0];
      if (!file) throw new Error("请选择 ZIP 文件");
      setModalBusy(true, "上传中...");
      toast("正在上传数据集，请勿重复点击");
      try {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("name", document.getElementById("uploadDatasetName").value.trim());
        const res = await fetch("/api/platform/datasets/upload", { method: "POST", headers: authHeaders(), body: fd });
        if (res.status === 401) {
          setModalBusy(false);
          redirectLogin();
          return;
        }
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "上传失败");
        modalBusy = false;
        toast("数据集上传成功");
        closeModal();
        loadPlatformDatasets();
      } catch (e) {
        setModalBusy(false);
        throw e;
      }
    },
  });
}

function openDeviceModal(deviceId = null) {
  const dev = deviceId ? platformState.devices.find((d) => d.id === deviceId) : null;
  openModal({
    title: dev ? "编辑设备" : "新建设备",
    wide: true,
    bodyHtml: `
      <div class="form-grid">
        <div><label>设备名称</label><input type="text" id="devName" value="${platformEsc(dev?.name || "")}" /></div>
        <div><label>设备类型</label><select id="devType"><option value="video" ${dev?.device_type === "video" ? "selected" : ""}>视频流</option><option value="image" ${dev?.device_type === "image" ? "selected" : ""}>图片流</option></select></div>
        <div class="full"><label>源地址 (RTSP/HTTP/摄像头索引 或 图片URL)</label><input type="text" id="devSource" value="${platformEsc(dev?.source || "")}" placeholder="rtsp://... 或 https://..." /></div>
        <div><label>图片轮询间隔 (秒)</label><input type="number" id="devPoll" min="1" value="${dev?.poll_interval || 5}" /></div>
        <div><label><input type="checkbox" id="devEnabled" ${dev?.enabled !== false ? "checked" : ""} /> 启用</label></div>
        ${dev ? `<p class="hint full">ROI 区域请在设备列表中点击「配置 ROI」进行可视化绘制（当前 ${(dev.roi || []).length} 个区域）。</p>` : `<p class="hint full">创建后可在列表中点击「配置 ROI」在预览画面上绘制检测区域。</p>`}
      </div>`,
    onConfirm: async () => {
      const body = {
        name: document.getElementById("devName").value.trim() || "未命名设备",
        device_type: document.getElementById("devType").value,
        source: document.getElementById("devSource").value.trim(),
        poll_interval: Number(document.getElementById("devPoll").value) || 5,
        enabled: document.getElementById("devEnabled").checked,
      };
      if (dev) {
        await api(`/api/platform/devices/${deviceId}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        toast("设备已更新");
      } else {
        await api("/api/platform/devices", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        toast("设备已创建");
      }
      closeModal();
      loadPlatformDevices();
    },
  });
}

const ROI_COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4"];
const roiEditorState = {
  deviceId: null,
  device: null,
  rois: [],
  drawing: null,
  activeIndex: -1,
  previewBlobUrl: null,
  loading: false,
};

function clamp01(v) {
  return Math.min(1, Math.max(0, v));
}

function normRectFromPixels(x, y, w, h, displayW, displayH) {
  return {
    x: clamp01(x / displayW),
    y: clamp01(y / displayH),
    w: clamp01(w / displayW),
    h: clamp01(h / displayH),
  };
}

function getRoiDisplaySize() {
  const img = document.getElementById("roiImage");
  if (!img) return { w: 0, h: 0 };
  return { w: img.clientWidth, h: img.clientHeight };
}

function renderRoiCanvas() {
  const canvas = document.getElementById("roiCanvas");
  const img = document.getElementById("roiImage");
  if (!canvas || !img || !img.complete || !img.naturalWidth) return;

  const { w: displayW, h: displayH } = getRoiDisplaySize();
  if (!displayW || !displayH) return;

  canvas.width = displayW;
  canvas.height = displayH;
  canvas.style.width = `${displayW}px`;
  canvas.style.height = `${displayH}px`;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, displayW, displayH);

  roiEditorState.rois.forEach((r, i) => {
    const px = r.x * displayW;
    const py = r.y * displayH;
    const pw = r.w * displayW;
    const ph = r.h * displayH;
    const color = ROI_COLORS[i % ROI_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = i === roiEditorState.activeIndex ? 3 : 2;
    ctx.strokeRect(px, py, pw, ph);
    ctx.fillStyle = color + "33";
    ctx.fillRect(px, py, pw, ph);
    ctx.fillStyle = color;
    ctx.font = "12px sans-serif";
    ctx.fillText(`区域 ${i + 1}`, px + 4, py + 14);
  });

  const d = roiEditorState.drawing;
  if (d) {
    const x = Math.min(d.startX, d.curX);
    const y = Math.min(d.startY, d.curY);
    const w = Math.abs(d.curX - d.startX);
    const h = Math.abs(d.curY - d.startY);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }
}

function renderRoiList() {
  const list = document.getElementById("roiList");
  const countEl = document.getElementById("roiCount");
  if (countEl) countEl.textContent = String(roiEditorState.rois.length);
  if (!list) return;
  if (!roiEditorState.rois.length) {
    list.innerHTML = '<p class="hint">暂无区域，在左侧画面拖拽绘制</p>';
    return;
  }
  list.innerHTML = roiEditorState.rois.map((r, i) => {
    const pct = (v) => (v * 100).toFixed(1);
    return `<div class="roi-list-item${i === roiEditorState.activeIndex ? " is-active" : ""}" data-roi-index="${i}">
      <div>
        <strong style="color:${ROI_COLORS[i % ROI_COLORS.length]}">区域 ${i + 1}</strong>
        <div class="roi-coords">x:${pct(r.x)}% y:${pct(r.y)}% w:${pct(r.w)}% h:${pct(r.h)}%</div>
      </div>
      <button type="button" class="btn btn-danger btn-sm" data-roi-del="${i}">删除</button>
    </div>`;
  }).join("");
  list.querySelectorAll("[data-roi-del]").forEach((btn) => {
    btn.onclick = (e) => {
      e.stopPropagation();
      const idx = Number(btn.dataset.roiDel);
      roiEditorState.rois.splice(idx, 1);
      roiEditorState.activeIndex = -1;
      renderRoiList();
      renderRoiCanvas();
    };
  });
  list.querySelectorAll("[data-roi-index]").forEach((row) => {
    row.onclick = () => {
      roiEditorState.activeIndex = Number(row.dataset.roiIndex);
      renderRoiList();
      renderRoiCanvas();
    };
  });
}

async function loadRoiPreview() {
  if (!roiEditorState.deviceId || roiEditorState.loading) return;
  roiEditorState.loading = true;
  const status = document.getElementById("roiStatus");
  if (status) status.textContent = "正在获取画面...";
  try {
    const url = authUrl(`/api/platform/devices/${roiEditorState.deviceId}/preview?t=${Date.now()}`);
    const res = await fetch(url, { headers: authHeaders() });
    if (res.status === 401) {
      redirectLogin();
      return;
    }
    if (!res.ok) {
      let detail = "获取画面失败";
      try {
        const data = await res.json();
        detail = data.detail || detail;
      } catch (_) {
        detail = await res.text() || detail;
      }
      throw new Error(detail);
    }
    if (roiEditorState.previewBlobUrl) URL.revokeObjectURL(roiEditorState.previewBlobUrl);
    const blob = await res.blob();
    roiEditorState.previewBlobUrl = URL.createObjectURL(blob);
    const img = document.getElementById("roiImage");
    if (img) {
      img.onload = () => {
        renderRoiCanvas();
        if (status) {
          status.textContent = roiEditorState.device?.device_type === "video"
            ? "视频截帧预览（可点击刷新获取最新帧）"
            : "图片流预览（可点击刷新获取最新图片）";
        }
      };
      img.onerror = () => {
        if (status) status.textContent = "画面加载失败";
      };
      img.src = roiEditorState.previewBlobUrl;
    }
  } catch (e) {
    toast(e.message, true);
    if (status) status.textContent = e.message;
  } finally {
    roiEditorState.loading = false;
  }
}

function openRoiEditor(deviceId) {
  const dev = platformState.devices.find((d) => d.id === deviceId);
  if (!dev) return toast("设备不存在", true);
  if (!(dev.source || "").trim()) return toast("请先配置设备源地址", true);

  roiEditorState.deviceId = deviceId;
  roiEditorState.device = dev;
  roiEditorState.rois = (dev.roi || []).map((r) => ({
    x: Number(r.x) || 0,
    y: Number(r.y) || 0,
    w: Number(r.w) || 0,
    h: Number(r.h) || 0,
  }));
  roiEditorState.drawing = null;
  roiEditorState.activeIndex = -1;

  const viewer = document.getElementById("roiViewer");
  const nameEl = document.getElementById("roiDeviceName");
  if (nameEl) nameEl.textContent = `${dev.name} · ROI 配置`;
  if (viewer) viewer.hidden = false;
  document.body.style.overflow = "hidden";

  renderRoiList();
  loadRoiPreview();
}

function closeRoiEditor() {
  const viewer = document.getElementById("roiViewer");
  if (viewer) viewer.hidden = true;
  document.body.style.overflow = "";
  if (roiEditorState.previewBlobUrl) {
    URL.revokeObjectURL(roiEditorState.previewBlobUrl);
    roiEditorState.previewBlobUrl = null;
  }
  roiEditorState.deviceId = null;
  roiEditorState.device = null;
  roiEditorState.drawing = null;
  const img = document.getElementById("roiImage");
  if (img) img.removeAttribute("src");
}

async function saveRoiEditor() {
  if (!roiEditorState.deviceId) return;
  try {
    await api(`/api/platform/devices/${roiEditorState.deviceId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ roi: roiEditorState.rois }),
    });
    toast("ROI 已保存");
    await loadPlatformDevices();
    closeRoiEditor();
  } catch (e) {
    toast(e.message, true);
  }
}

function roiCanvasPoint(evt) {
  const canvas = document.getElementById("roiCanvas");
  if (!canvas) return { x: 0, y: 0 };
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.min(Math.max(0, evt.clientX - rect.left), rect.width),
    y: Math.min(Math.max(0, evt.clientY - rect.top), rect.height),
  };
}

function initRoiEditor() {
  const canvas = document.getElementById("roiCanvas");
  document.getElementById("roiClose")?.addEventListener("click", closeRoiEditor);
  document.getElementById("roiRefreshPreview")?.addEventListener("click", () => loadRoiPreview());
  document.getElementById("roiClearAll")?.addEventListener("click", () => {
    if (roiEditorState.rois.length && !confirm("确定清除全部 ROI 区域？")) return;
    roiEditorState.rois = [];
    roiEditorState.activeIndex = -1;
    renderRoiList();
    renderRoiCanvas();
  });
  document.getElementById("roiSave")?.addEventListener("click", () => saveRoiEditor());

  if (canvas) {
    canvas.addEventListener("mousedown", (evt) => {
      if (roiEditorState.loading) return;
      const p = roiCanvasPoint(evt);
      roiEditorState.drawing = { startX: p.x, startY: p.y, curX: p.x, curY: p.y };
      roiEditorState.activeIndex = -1;
      renderRoiList();
    });
    canvas.addEventListener("mousemove", (evt) => {
      if (!roiEditorState.drawing) return;
      const p = roiCanvasPoint(evt);
      roiEditorState.drawing.curX = p.x;
      roiEditorState.drawing.curY = p.y;
      renderRoiCanvas();
    });
    const finishDraw = () => {
      if (!roiEditorState.drawing) return;
      const d = roiEditorState.drawing;
      const x = Math.min(d.startX, d.curX);
      const y = Math.min(d.startY, d.curY);
      const w = Math.abs(d.curX - d.startX);
      const h = Math.abs(d.curY - d.startY);
      roiEditorState.drawing = null;
      const { w: displayW, h: displayH } = getRoiDisplaySize();
      if (displayW && displayH && w >= 8 && h >= 8) {
        const norm = normRectFromPixels(x, y, w, h, displayW, displayH);
        if (norm.w >= 0.01 && norm.h >= 0.01) {
          roiEditorState.rois.push(norm);
          roiEditorState.activeIndex = roiEditorState.rois.length - 1;
        }
      }
      renderRoiList();
      renderRoiCanvas();
    };
    canvas.addEventListener("mouseup", finishDraw);
    canvas.addEventListener("mouseleave", finishDraw);
  }

  window.addEventListener("resize", () => {
    if (!document.getElementById("roiViewer")?.hidden) renderRoiCanvas();
  });
  document.addEventListener("keydown", (e) => {
    const viewer = document.getElementById("roiViewer");
    if (viewer?.hidden) return;
    if (e.key === "Escape") closeRoiEditor();
  });
}

async function createTrainingJobFromPanel() {
  const modelId = document.getElementById("modalTrainModelSelect")?.value;
  const datasetId = document.getElementById("modalTrainDatasetSelect")?.value;
  if (!modelId || !datasetId) throw new Error("请选择模型和数据集");
  await api("/api/platform/training-jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: document.getElementById("modalTrainJobName")?.value.trim() || undefined,
      model_id: modelId,
      dataset_id: datasetId,
      epochs: Number(document.getElementById("modalTrainEpochsInput")?.value || 50),
      batch_size: Number(document.getElementById("modalTrainBatchInput")?.value || 4),
      grad_accum_steps: Number(document.getElementById("modalTrainGradInput")?.value || 4),
      lr: Number(document.getElementById("modalTrainLrInput")?.value || 0.0001),
      gpu_ids: [...selectedTrainGpus].sort((a, b) => a - b),
    }),
  });
  toast("训练任务已创建");
  loadTrainingJobs();
}

function openCreateTrainJobModal() {
  const cfg = lastStatus?.config?.training || {};
  selectedTrainGpus = new Set(cfg.gpu_ids || [0]);
  openModal({
    title: "新建训练任务",
    wide: true,
    confirmText: "创建",
    bodyHtml: `
      <div class="form-grid">
        <div class="full"><label>任务名称（可选）</label><input type="text" id="modalTrainJobName" placeholder="smoke-v2" /></div>
        <div class="full"><label>基础模型</label><select id="modalTrainModelSelect"><option value="">— 选择模型 —</option></select></div>
        <div class="full"><label>数据集</label><select id="modalTrainDatasetSelect"><option value="">— 选择数据集 —</option></select></div>
        <div class="full"><label>训练 GPU（多选 = 多卡 DDP）</label><div id="modalTrainGpuChips" class="gpu-chips"></div></div>
        <p class="hint full" id="modalTrainGpuHint">已选 1 张卡：单卡训练</p>
        <div><label>Epochs</label><input type="number" id="modalTrainEpochsInput" min="1" value="${cfg.epochs || 50}" /></div>
        <div><label>Batch Size</label><input type="number" id="modalTrainBatchInput" min="1" value="${cfg.batch_size || 4}" /></div>
        <div><label>Grad Accum</label><input type="number" id="modalTrainGradInput" min="1" value="${cfg.grad_accum_steps || 4}" /></div>
        <div><label>Learning Rate</label><input type="number" id="modalTrainLrInput" step="0.00001" value="${cfg.lr || 0.0001}" /></div>
      </div>`,
    onConfirm: async () => {
      await createTrainingJobFromPanel();
      closeModal();
    },
  });
  refreshModelSelects();
  refreshDatasetSelects();
  buildGpuChips($("#modalTrainGpuChips"), cachedGpus, selectedTrainGpus, () => {
    const hint = $("#modalTrainGpuHint");
    const n = selectedTrainGpus.size;
    if (hint) {
      hint.textContent = n <= 1
        ? `已选 ${n} 张卡：单卡训练（物理 GPU ${[...selectedTrainGpus].join(", ")}）`
        : `已选 ${n} 张卡：多卡 DDP 训练（物理 GPU ${[...selectedTrainGpus].sort((a, b) => a - b).join(", ")}）`;
    }
  });
}

let trainLogPollTimer = null;

function openTrainJobLogModal(jobId) {
  const job = platformState.trainingJobs.find((j) => j.id === jobId);
  openModal({
    title: `训练日志 · ${job?.name || jobId}`,
    wide: true,
    confirmText: "关闭",
    bodyHtml: `
      <p class="hint" id="modalTrainLogMeta">加载中...</p>
      <pre id="modalTrainLogContent" class="log-box" style="max-height:60vh;overflow:auto">加载中...</pre>`,
    onConfirm: async () => {
      if (trainLogPollTimer) {
        clearInterval(trainLogPollTimer);
        trainLogPollTimer = null;
      }
      closeModal();
    },
  });
  const confirmBtn = $("#modalConfirm");
  if (confirmBtn) confirmBtn.textContent = "关闭";

  const loadLog = async () => {
    try {
      const data = await api(`/api/platform/training-jobs/${jobId}/log`);
      const meta = $("#modalTrainLogMeta");
      const pre = $("#modalTrainLogContent");
      if (meta) {
        meta.textContent = `状态: ${data.state}${data.live ? " · 实时更新中" : ""} · ${data.message || ""}`;
      }
      if (pre) pre.textContent = data.content || "暂无日志";
      return data.live;
    } catch (e) {
      toast(e.message, true);
      return false;
    }
  };
  loadLog().then((live) => {
    if (trainLogPollTimer) clearInterval(trainLogPollTimer);
    if (!live) return;
    trainLogPollTimer = setInterval(() => {
      if (!document.getElementById("modalTrainLogContent")) {
        clearInterval(trainLogPollTimer);
        trainLogPollTimer = null;
        return;
      }
      loadLog();
    }, 3000);
  });
}

function initPlatformPanels() {
  tablePageState.model = { page: 1, pageSize: 10 };
  tablePageState.dataset = { page: 1, pageSize: 10 };
  tablePageState.device = { page: 1, pageSize: 10 };
  tablePageState.alert = { page: 1, pageSize: 12 };
  tablePageState.trainJob = { page: 1, pageSize: 10 };

  document.getElementById("btnUploadModel")?.addEventListener("click", openUploadModelModal);
  document.getElementById("btnUploadDataset")?.addEventListener("click", openUploadDatasetModal);
  document.getElementById("btnAddDevice")?.addEventListener("click", () => openDeviceModal());
  document.getElementById("btnAddTrainJob")?.addEventListener("click", openCreateTrainJobModal);
  document.getElementById("modelSearch")?.addEventListener("input", () => { platformPageState.model.page = 1; renderModelTable(); });
  document.getElementById("datasetSearch")?.addEventListener("input", () => { platformPageState.dataset.page = 1; renderDatasetTable(); });
  bindAuditFilterEvents("device", "device", renderDeviceTable);
  bindAuditFilterEvents("trainJob", "trainJob", renderTrainingJobTable);
  document.getElementById("alertDeviceFilter")?.addEventListener("change", (e) => {
    const deviceId = e.target.value;
    const p = platformPageState.alert.page;
    api(`/api/platform/alerts?page=${p}&page_size=12${deviceId ? "&device_id=" + deviceId : ""}`).then(renderAlertGrid).catch(() => {});
  });
  document.getElementById("btnRefreshAlerts")?.addEventListener("click", () => loadPlatformAlerts(1));
  initDatasetViewer();
  initRoiEditor();
}

async function refreshPlatformData(tabKey) {
  if (tabKey === "models" || tabKey === "gpu" || tabKey === "train") await loadPlatformModels();
  if (tabKey === "datasets" || tabKey === "train") await loadPlatformDatasets();
  if (tabKey === "devices" || tabKey === "gpu") await loadPlatformDevices();
  if (tabKey === "alerts") await loadPlatformAlerts();
  if (tabKey === "train") await loadTrainingJobs();
  if (tabKey === "gpu") await loadAnalysisStatus();
}
