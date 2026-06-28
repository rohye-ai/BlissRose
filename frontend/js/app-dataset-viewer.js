/** Dataset annotation workbench — performance & UX optimized */
const DV_LABEL_COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4"];

const datasetViewerState = {
  datasetId: null,
  datasetName: "",
  split: "train",
  statusFilter: "all",
  fileSearch: "",
  filePage: 1,
  filePageSize: 50,
  totalFiles: 0,
  totalPages: 1,
  images: [],
  index: 0,
  tool: "rect",
  drawing: null,
  selectedClassId: 0,
  selectedAnnIdx: -1,
  annotations: [],
  savedSnapshot: "[]",
  classNames: [],
  reviewStatus: "draft",
  stats: {},
  loading: false,
  dirty: false,
  zoom: 1,
  panX: 0,
  panY: 0,
  isPanning: false,
  panStart: null,
  undoStack: [],
  redoStack: [],
  labelCache: new Map(),
  imagePreload: new Map(),
  loadSeq: 0,
  overlayRAF: 0,
  fileListBound: false,
  fileSearchTimer: null,
  fileListLoading: false,
};

function datasetPreviewUrl(datasetId, itemPath) {
  if (!datasetId || !itemPath) return "";
  return authUrl(`/api/platform/datasets/${datasetId}/file?path=${encodeURIComponent(itemPath)}`);
}

function dvAnnotationsSnapshot(anns) {
  return JSON.stringify(anns || []);
}

function dvMarkSaved() {
  datasetViewerState.savedSnapshot = dvAnnotationsSnapshot(datasetViewerState.annotations);
  datasetViewerState.dirty = false;
  updateDvUnsavedBadge();
}

function dvMarkDirty() {
  datasetViewerState.dirty = dvAnnotationsSnapshot(datasetViewerState.annotations) !== datasetViewerState.savedSnapshot;
  updateDvUnsavedBadge();
}

function updateDvUnsavedBadge() {
  const el = document.getElementById("dvUnsavedBadge");
  if (el) el.hidden = !datasetViewerState.dirty;
}

function dvPushUndo() {
  datasetViewerState.undoStack.push(dvAnnotationsSnapshot(datasetViewerState.annotations));
  if (datasetViewerState.undoStack.length > 60) datasetViewerState.undoStack.shift();
  datasetViewerState.redoStack = [];
  dvMarkDirty();
}

function dvUndo() {
  if (!datasetViewerState.undoStack.length) return;
  datasetViewerState.redoStack.push(dvAnnotationsSnapshot(datasetViewerState.annotations));
  datasetViewerState.annotations = JSON.parse(datasetViewerState.undoStack.pop());
  datasetViewerState.selectedAnnIdx = -1;
  dvMarkDirty();
  renderDatasetViewerOverlay();
  renderDatasetViewerLabelList();
}

function dvRedo() {
  if (!datasetViewerState.redoStack.length) return;
  datasetViewerState.undoStack.push(dvAnnotationsSnapshot(datasetViewerState.annotations));
  datasetViewerState.annotations = JSON.parse(datasetViewerState.redoStack.pop());
  datasetViewerState.selectedAnnIdx = -1;
  dvMarkDirty();
  renderDatasetViewerOverlay();
  renderDatasetViewerLabelList();
}

function dvInvalidateLabelCache(path) {
  if (path) datasetViewerState.labelCache.delete(path);
  else datasetViewerState.labelCache.clear();
}

function dvSetLoading(on) {
  datasetViewerState.loading = on;
  const el = document.getElementById("dvLoading");
  if (!el) return;
  el.hidden = !on;
  el.style.display = on ? "flex" : "none";
}

function dvGlobalIndex() {
  return (datasetViewerState.filePage - 1) * datasetViewerState.filePageSize + datasetViewerState.index;
}

function dvSetImageLoadState(state, hint = "") {
  const viewport = document.getElementById("dvViewport");
  const imgEl = document.getElementById("dvImage");
  const placeholder = document.getElementById("dvImagePlaceholder");
  const placeholderText = document.getElementById("dvPlaceholderText");
  if (placeholderText) {
    placeholderText.textContent = hint || (state === "loading" ? "加载中..." : "暂无图片");
  }
  if (state === "loaded") {
    viewport?.classList.remove("is-loading");
    if (placeholder) placeholder.hidden = true;
    if (imgEl) imgEl.hidden = false;
    dvSetLoading(false);
    return;
  }
  viewport?.classList.add("is-loading");
  if (placeholder) placeholder.hidden = false;
  if (imgEl) {
    imgEl.hidden = true;
    imgEl.removeAttribute("src");
  }
  dvSetLoading(state === "loading");
}

function dvResetView() {
  datasetViewerState.zoom = 1;
  datasetViewerState.panX = 0;
  datasetViewerState.panY = 0;
  applyDvTransform();
}

function applyDvTransform() {
  const layer = document.getElementById("dvTransformLayer");
  const zoomEl = document.getElementById("dvZoomLabel");
  if (layer) {
    layer.style.transform = `translate(${datasetViewerState.panX}px, ${datasetViewerState.panY}px) scale(${datasetViewerState.zoom})`;
  }
  if (zoomEl) zoomEl.textContent = `${Math.round(datasetViewerState.zoom * 100)}%`;
  scheduleDatasetViewerOverlay();
}

function dvZoomBy(delta, centerX, centerY) {
  const old = datasetViewerState.zoom;
  const next = Math.min(4, Math.max(0.25, old + delta));
  if (next === old) return;
  const viewport = document.getElementById("dvViewport");
  if (viewport && centerX != null) {
    const rect = viewport.getBoundingClientRect();
    const cx = centerX - rect.left - rect.width / 2;
    const cy = centerY - rect.top - rect.height / 2;
    const ratio = next / old;
    datasetViewerState.panX = cx - (cx - datasetViewerState.panX) * ratio;
    datasetViewerState.panY = cy - (cy - datasetViewerState.panY) * ratio;
  }
  datasetViewerState.zoom = next;
  applyDvTransform();
}

function updateDatasetViewerReviewToolbar() {
  const rs = datasetViewerState.reviewStatus;
  const submitBtn = document.getElementById("dvSubmitReview");
  const approveBtn = document.getElementById("dvApproveReview");
  const rejectBtn = document.getElementById("dvRejectReview");
  const imgApprove = document.getElementById("dvImageApprove");
  const imgReject = document.getElementById("dvImageReject");
  if (submitBtn) submitBtn.hidden = rs === "approved" || rs === "pending_review";
  if (approveBtn) approveBtn.hidden = rs !== "pending_review";
  if (rejectBtn) rejectBtn.hidden = rs !== "pending_review";
  const showImgReview = rs === "pending_review";
  if (imgApprove) imgApprove.hidden = !showImgReview;
  if (imgReject) imgReject.hidden = !showImgReview;
}

function updateDvProgressUI() {
  const st = datasetViewerState.stats;
  const total = st.total_count ?? datasetViewerState.images.length;
  const labeled = st.labeled_count ?? 0;
  const progress = document.getElementById("dvProgressText");
  const bar = document.getElementById("dvProgressBar");
  const pct = total > 0 ? Math.round((labeled / total) * 100) : 0;
  if (progress) progress.textContent = `已标注 ${labeled} / ${total}（${pct}%）`;
  if (bar) bar.style.width = `${pct}%`;
}

function getFilteredDvImages() {
  return datasetViewerState.images;
}

function dvSetFileListLoading(on) {
  datasetViewerState.fileListLoading = on;
  document.getElementById("dvFileListWrap")?.classList.toggle("is-loading", on);
  const loading = document.getElementById("dvFileListLoading");
  if (loading) loading.hidden = !on;
  document.getElementById("dvFilePagination")?.classList.toggle("is-loading", on);
}

function renderDvFilePagination() {
  const el = document.getElementById("dvFilePagination");
  if (!el) return;
  const total = datasetViewerState.totalFiles;
  const loading = datasetViewerState.fileListLoading;
  if (!total && !loading) {
    el.innerHTML = "";
    el.hidden = true;
    return;
  }
  el.hidden = false;
  const page = datasetViewerState.filePage;
  const pageSize = datasetViewerState.filePageSize;
  const totalPages = datasetViewerState.totalPages || 1;
  const rangeStart = total ? (page - 1) * pageSize + 1 : 0;
  const rangeEnd = total ? Math.min(page * pageSize, total) : 0;
  const loadingHint = loading ? ' · <span class="dv-pg-loading">加载中...</span>' : "";
  el.innerHTML = `
    <div class="pagination-info">${total ? `${rangeStart}-${rangeEnd} / 共 ${total}` : "加载列表..."}${loadingHint}</div>
    <div class="pagination-controls">
      <button type="button" class="btn btn-secondary btn-sm" data-dv-pg="prev" ${page <= 1 || loading ? "disabled" : ""}>上一页</button>
      <span class="pagination-pages">${loading ? "…" : `${page}/${totalPages}`}</span>
      <button type="button" class="btn btn-secondary btn-sm" data-dv-pg="next" ${page >= totalPages || loading ? "disabled" : ""}>下一页</button>
      <select class="pagination-size" aria-label="每页条数" ${loading ? "disabled" : ""}>
        <option value="30" ${pageSize === 30 ? "selected" : ""}>30</option>
        <option value="50" ${pageSize === 50 ? "selected" : ""}>50</option>
        <option value="100" ${pageSize === 100 ? "selected" : ""}>100</option>
      </select>
    </div>`;
  el.querySelector('[data-dv-pg="prev"]')?.addEventListener("click", async () => {
    if (datasetViewerState.fileListLoading || datasetViewerState.filePage <= 1) return;
    if (!(await confirmLeaveIfDirty())) return;
    datasetViewerState.filePage -= 1;
    await loadDatasetViewerImages(true, 0);
  });
  el.querySelector('[data-dv-pg="next"]')?.addEventListener("click", async () => {
    if (datasetViewerState.fileListLoading || datasetViewerState.filePage >= datasetViewerState.totalPages) return;
    if (!(await confirmLeaveIfDirty())) return;
    datasetViewerState.filePage += 1;
    await loadDatasetViewerImages(true, 0);
  });
  el.querySelector(".pagination-size")?.addEventListener("change", async (e) => {
    if (datasetViewerState.fileListLoading) return;
    if (!(await confirmLeaveIfDirty())) {
      renderDvFilePagination();
      return;
    }
    datasetViewerState.filePageSize = Number(e.target.value) || 50;
    datasetViewerState.filePage = 1;
    await loadDatasetViewerImages(false, 0);
  });
}

function renderDatasetFileList() {
  const list = document.getElementById("dvFileList");
  if (!list) return;
  const { index } = datasetViewerState;
  const visible = getFilteredDvImages();
  if (!visible.length) {
    list.innerHTML = '<p class="hint dv-empty-hint">暂无匹配图片</p>';
    return;
  }
  const frag = document.createDocumentFragment();
  visible.forEach((item) => {
    const i = datasetViewerState.images.indexOf(item);
    if (i < 0) return;
    const st = item.annotate_status || (item.has_labels ? "labeled" : "unlabeled");
    const label = item.annotate_status_label || (st === "labeled" ? "已标注" : "未标注");
    const row = document.createElement("div");
    row.className = `dv-file-item${i === index ? " active" : ""}`;
    row.dataset.fileIdx = String(i);
    row.innerHTML = `<span class="name" title="${platformEsc(item.name)}">${platformEsc(item.name)}</span>
      <span class="dv-status-badge ${annotateStatusClass(st)}">${platformEsc(label)}</span>`;
    frag.appendChild(row);
  });
  list.innerHTML = "";
  list.appendChild(frag);
  const active = list.querySelector(".dv-file-item.active");
  if (active) active.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function bindDatasetFileListEvents() {
  if (datasetViewerState.fileListBound) return;
  const list = document.getElementById("dvFileList");
  if (!list) return;
  list.addEventListener("click", async (e) => {
    const row = e.target.closest("[data-file-idx]");
    if (!row) return;
    const idx = Number(row.dataset.fileIdx);
    if (Number.isNaN(idx)) return;
    await navigateDatasetViewerImage(idx);
  });
  datasetViewerState.fileListBound = true;
}

function renderDatasetClassList() {
  const list = document.getElementById("dvClassList");
  const classSel = document.getElementById("dvClassSelect");
  const names = datasetViewerState.classNames.length ? datasetViewerState.classNames : ["class_0"];
  if (list) {
    list.innerHTML = names.map((n, i) => {
      const color = DV_LABEL_COLORS[i % DV_LABEL_COLORS.length];
      const hotkey = i < 9 ? `<span class="dv-hotkey">${i + 1}</span>` : "";
      return `<div class="dv-class-item${i === datasetViewerState.selectedClassId ? " active" : ""}" data-class-id="${i}">
        <span class="dv-class-dot" style="background:${color}"></span>
        <span class="dv-class-name">${platformEsc(n)}</span>${hotkey}
      </div>`;
    }).join("");
    list.querySelectorAll("[data-class-id]").forEach((el) => {
      el.addEventListener("click", () => {
        datasetViewerState.selectedClassId = Number(el.getAttribute("data-class-id")) || 0;
        renderDatasetClassList();
        if (classSel) classSel.value = String(datasetViewerState.selectedClassId);
        setDvTool("rect");
      });
    });
  }
  if (classSel) {
    classSel.innerHTML = names.map((n, i) =>
      `<option value="${i}">${platformEsc(n)}</option>`
    ).join("");
    classSel.value = String(datasetViewerState.selectedClassId);
  }
}

function setDvTool(tool) {
  datasetViewerState.tool = tool;
  document.getElementById("dvToolSelect")?.classList.toggle("active", tool === "select");
  document.getElementById("dvToolRect")?.classList.toggle("active", tool === "rect");
  const canvas = document.getElementById("dvCanvas");
  if (canvas) {
    canvas.style.cursor = tool === "rect" ? "crosshair" : "default";
  }
}

async function confirmLeaveIfDirty() {
  if (!datasetViewerState.dirty) return true;
  return confirm("当前图片标注未保存，是否放弃修改？");
}

async function openDatasetViewer(datasetId, datasetName = "") {
  datasetViewerState.datasetId = datasetId;
  datasetViewerState.datasetName = datasetName;
  datasetViewerState.split = "train";
  datasetViewerState.statusFilter = "all";
  datasetViewerState.fileSearch = "";
  datasetViewerState.filePage = 1;
  datasetViewerState.filePageSize = 50;
  datasetViewerState.index = 0;
  datasetViewerState.undoStack = [];
  datasetViewerState.redoStack = [];
  datasetViewerState.labelCache.clear();
  datasetViewerState.imagePreload.clear();
  dvResetView();
  setDvTool("rect");
  bindDatasetFileListEvents();
  const viewer = document.getElementById("datasetViewer");
  const splitSel = document.getElementById("dvSplitSelect");
  const filterSel = document.getElementById("dvStatusFilter");
  const searchEl = document.getElementById("dvFileSearch");
  if (splitSel) splitSel.value = "train";
  if (filterSel) filterSel.value = "all";
  if (searchEl) searchEl.value = "";
  if (viewer) viewer.hidden = false;
  document.body.style.overflow = "hidden";
  dvSetImageLoadState("loading");
  await loadDatasetViewerImages(false, 0);
}

function closeDatasetViewer() {
  if (datasetViewerState.dirty && !confirm("有未保存的标注，确定关闭？")) return;
  const viewer = document.getElementById("datasetViewer");
  if (viewer) viewer.hidden = true;
  document.body.style.overflow = "";
  datasetViewerState.images = [];
  datasetViewerState.annotations = [];
  datasetViewerState.loadSeq += 1;
  dvSetImageLoadState("empty");
  dvSetFileListLoading(false);
}

async function loadDatasetViewerImages(keepIndex = false, targetIndex = 0) {
  const { datasetId, split, statusFilter, filePage, filePageSize, fileSearch } = datasetViewerState;
  if (!datasetId) return;
  if (!keepIndex && !(await confirmLeaveIfDirty())) return;
  datasetViewerState.loading = true;
  dvSetFileListLoading(true);
  renderDvFilePagination();
  const meta = document.getElementById("dvMeta");
  if (meta) meta.textContent = "加载列表...";
  try {
    const q = new URLSearchParams({
      split,
      status: statusFilter,
      page: String(filePage),
      page_size: String(filePageSize),
    });
    if (fileSearch.trim()) q.set("search", fileSearch.trim());
    const data = await api(`/api/platform/datasets/${datasetId}/images?${q}`);
    if (!keepIndex) datasetViewerState.index = 0;
    datasetViewerState.images = data.items || [];
    datasetViewerState.totalFiles = data.total ?? datasetViewerState.images.length;
    datasetViewerState.totalPages = data.total_pages ?? 1;
    datasetViewerState.filePage = data.page ?? filePage;
    datasetViewerState.datasetName = data.dataset_name || datasetViewerState.datasetName;
    datasetViewerState.classNames = data.class_names || [];
    datasetViewerState.reviewStatus = data.review_status || "draft";
    datasetViewerState.stats = data.stats || {};
    const idx = keepIndex
      ? Math.min(
          targetIndex ?? datasetViewerState.index,
          Math.max(0, datasetViewerState.images.length - 1)
        )
      : 0;
    datasetViewerState.index = idx;
    datasetViewerState.labelCache.clear();
    const nameEl = document.getElementById("dvDatasetName");
    if (nameEl) nameEl.textContent = datasetViewerState.datasetName || "数据集标注";
    if (meta) {
      meta.textContent = `${split} · ${reviewStatusLabel(datasetViewerState.reviewStatus)}${data.train_ready ? " · 可用于训练" : " · 审核通过后可训练"}`;
    }
    updateDvProgressUI();
    updateDatasetViewerReviewToolbar();
    renderDatasetFileList();
    renderDvFilePagination();
    renderDatasetClassList();
    if (!datasetViewerState.images.length) {
      document.getElementById("dvImageName").textContent = "该筛选下暂无图片";
      updateDatasetViewerNav();
      dvSetImageLoadState("empty");
      return;
    }
    await showDatasetViewerImage(datasetViewerState.index);
  } catch (e) {
    toast(e.message, true);
    closeDatasetViewer();
  } finally {
    datasetViewerState.loading = false;
    dvSetFileListLoading(false);
    renderDvFilePagination();
  }
}

async function navigateToImagePath(path, pageHint, indexHint) {
  if (!(await confirmLeaveIfDirty())) return;
  const localIdx = datasetViewerState.images.findIndex((i) => i.path === path);
  if (localIdx >= 0) {
    await showDatasetViewerImage(localIdx);
    return;
  }
  if (pageHint) {
    datasetViewerState.filePage = pageHint;
    await loadDatasetViewerImages(true, indexHint ?? 0);
  }
}

function prefetchDatasetImages(centerIdx) {
  const { images, datasetId } = datasetViewerState;
  [-1, 1, 2].forEach((off) => {
    const idx = centerIdx + off;
    if (idx < 0 || idx >= images.length) return;
    const path = images[idx].path;
    if (datasetViewerState.imagePreload.has(path)) return;
    const img = new Image();
    img.decoding = "async";
    img.src = datasetPreviewUrl(datasetId, path);
    datasetViewerState.imagePreload.set(path, img);
  });
}

async function fetchLabelsForPath(path) {
  if (datasetViewerState.labelCache.has(path)) {
    return datasetViewerState.labelCache.get(path);
  }
  const data = await api(
    `/api/platform/datasets/${datasetViewerState.datasetId}/labels?path=${encodeURIComponent(path)}`
  );
  datasetViewerState.labelCache.set(path, data);
  return data;
}

async function navigateDatasetViewerImage(index) {
  if (datasetViewerState.loading) return;
  if (index === datasetViewerState.index) return;
  if (!(await confirmLeaveIfDirty())) return;
  await showDatasetViewerImage(index);
}

async function showDatasetViewerImage(index) {
  const images = datasetViewerState.images;
  if (!images.length || index < 0 || index >= images.length) return;
  const seq = ++datasetViewerState.loadSeq;
  datasetViewerState.index = index;
  const item = images[index];
  item._wasLabeled = !!(item.has_labels || ["labeled", "approved", "rejected"].includes(item.annotate_status));
  const imgEl = document.getElementById("dvImage");
  const nameEl = document.getElementById("dvImageName");
  const statusEl = document.getElementById("dvImageStatus");
  const src = datasetPreviewUrl(datasetViewerState.datasetId, item.path);
  const globalIdx = dvGlobalIndex();

  datasetViewerState.selectedAnnIdx = -1;
  datasetViewerState.drawing = null;
  datasetViewerState.undoStack = [];
  datasetViewerState.redoStack = [];
  dvSetImageLoadState("loading", item.name);

  const finishLoad = () => {
    if (seq !== datasetViewerState.loadSeq) return;
    dvSetImageLoadState("loaded");
    if (statusEl) {
      statusEl.innerHTML = `<div><strong>${platformEsc(item.annotate_status_label || "")}</strong></div>
        <div>对象 ${datasetViewerState.annotations.length} 个</div>`;
    }
    renderDatasetViewerOverlay();
    renderDatasetViewerLabelList();
  };

  try {
    updateDatasetViewerNav();
    renderDatasetFileList();

    if (nameEl) {
      nameEl.textContent = `${globalIdx + 1} / ${datasetViewerState.totalFiles} · ${item.name}`;
    }
    if (statusEl) {
      const stLabel = item.annotate_status_label || reviewStatusLabel(item.annotate_status);
      statusEl.hidden = false;
      statusEl.innerHTML = `<div><strong>${platformEsc(stLabel)}</strong></div>
        <div>对象 ${item.has_labels ? "…" : "0"} 个</div>`;
    }

    let annotations = [];
    if (item.has_labels || item.annotate_status !== "unlabeled") {
      try {
        const labelData = await fetchLabelsForPath(item.path);
        if (seq !== datasetViewerState.loadSeq) return;
        annotations = labelData.annotations || [];
      } catch (_) {
        annotations = [];
      }
    }
    if (seq !== datasetViewerState.loadSeq) return;

    datasetViewerState.annotations = annotations;
    dvMarkSaved();
    prefetchDatasetImages(index);

    if (!imgEl) {
      finishLoad();
      return;
    }

    imgEl.onerror = () => {
      if (seq !== datasetViewerState.loadSeq) return;
      dvSetImageLoadState("empty", "图片加载失败");
      toast("图片加载失败", true);
    };

    const bindAndLoad = () => {
      imgEl.onload = () => finishLoad();
      imgEl.decoding = "async";
      imgEl.dataset.currentPath = item.path;
      const prevSrc = imgEl.getAttribute("src") || "";
      if (prevSrc === src && imgEl.complete && imgEl.naturalWidth) {
        finishLoad();
        return;
      }
      imgEl.src = src;
      if (imgEl.complete && imgEl.naturalWidth) finishLoad();
    };

    bindAndLoad();
  } catch (e) {
    if (seq === datasetViewerState.loadSeq) {
      dvSetImageLoadState("empty", "加载失败");
      toast(e.message || "加载图片失败", true);
    }
  }
}

function updateDatasetViewerNav() {
  const { totalFiles } = datasetViewerState;
  const globalIdx = dvGlobalIndex();
  const prev = document.getElementById("dvPrev");
  const next = document.getElementById("dvNext");
  if (prev) prev.disabled = globalIdx <= 0;
  if (next) next.disabled = !totalFiles || globalIdx >= totalFiles - 1;
}

function scheduleDatasetViewerOverlay() {
  if (datasetViewerState.overlayRAF) return;
  datasetViewerState.overlayRAF = requestAnimationFrame(() => {
    datasetViewerState.overlayRAF = 0;
    renderDatasetViewerOverlay();
  });
}

function dvDisplayMetrics() {
  const img = document.getElementById("dvImage");
  if (!img || !img.complete || !img.naturalWidth) return null;
  const displayW = img.clientWidth;
  const displayH = img.clientHeight;
  if (!displayW || !displayH) return null;
  return { img, displayW, displayH };
}

function dvCanvasPoint(evt) {
  const canvas = document.getElementById("dvCanvas");
  if (!canvas || !canvas.width) return { x: 0, y: 0 };
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return { x: 0, y: 0 };
  return {
    x: Math.min(Math.max(0, ((evt.clientX - rect.left) / rect.width) * canvas.width), canvas.width),
    y: Math.min(Math.max(0, ((evt.clientY - rect.top) / rect.height) * canvas.height), canvas.height),
  };
}

function pixelRectToYolo(x, y, w, h, displayW, displayH) {
  return {
    cx: (x + w / 2) / displayW,
    cy: (y + h / 2) / displayH,
    w: w / displayW,
    h: h / displayH,
  };
}

function yoloToPixelRect(ann, displayW, displayH) {
  const bw = ann.w * displayW;
  const bh = ann.h * displayH;
  const cx = ann.cx * displayW;
  const cy = ann.cy * displayH;
  return { x: cx - bw / 2, y: cy - bh / 2, w: bw, h: bh };
}

function hitTestAnnotation(x, y) {
  const m = dvDisplayMetrics();
  if (!m) return -1;
  const { displayW, displayH } = m;
  for (let i = datasetViewerState.annotations.length - 1; i >= 0; i -= 1) {
    const r = yoloToPixelRect(datasetViewerState.annotations[i], displayW, displayH);
    if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return i;
  }
  return -1;
}

function renderDatasetViewerOverlay() {
  const canvas = document.getElementById("dvCanvas");
  const m = dvDisplayMetrics();
  if (!canvas || !m) return;
  const { displayW, displayH } = m;

  canvas.width = displayW;
  canvas.height = displayH;
  canvas.style.width = `${displayW}px`;
  canvas.style.height = `${displayH}px`;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, displayW, displayH);

  datasetViewerState.annotations.forEach((ann, idx) => {
    const r = yoloToPixelRect(ann, displayW, displayH);
    const color = DV_LABEL_COLORS[ann.class_id % DV_LABEL_COLORS.length];
    const selected = idx === datasetViewerState.selectedAnnIdx;
    ctx.strokeStyle = color;
    ctx.lineWidth = selected ? 3 : 2;
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    if (selected) {
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = "#fff";
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.setLineDash([]);
    }
    ctx.fillStyle = color;
    ctx.font = "12px sans-serif";
    const label = ann.class_name;
    const tw = ctx.measureText(label).width + 8;
    ctx.fillRect(r.x, Math.max(0, r.y - 18), tw, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, r.x + 4, Math.max(12, r.y - 5));
  });

  if (datasetViewerState.drawing) {
    const d = datasetViewerState.drawing;
    const x = Math.min(d.startX, d.curX);
    const y = Math.min(d.startY, d.curY);
    const w = Math.abs(d.curX - d.startX);
    const h = Math.abs(d.curY - d.startY);
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }
}

function renderDatasetViewerLabelList() {
  const list = document.getElementById("dvLabelList");
  if (!list) return;
  list.innerHTML = datasetViewerState.annotations.length
    ? datasetViewerState.annotations.map((ann, idx) => {
        const color = DV_LABEL_COLORS[ann.class_id % DV_LABEL_COLORS.length];
        const sel = idx === datasetViewerState.selectedAnnIdx ? " dv-layer-selected" : "";
        return `<div class="dv-layer-item${sel}" data-ann-idx="${idx}">
          <span class="dv-layer-label"><span class="dv-class-dot" style="background:${color}"></span>${platformEsc(ann.class_name)} #${idx + 1}</span>
          <button type="button" class="btn btn-danger btn-sm" data-ann-del="${idx}">删</button>
        </div>`;
      }).join("")
    : '<p class="hint">暂无对象 · 矩形工具拖拽画框</p>';
  list.querySelectorAll("[data-ann-idx]").forEach((el) => {
    el.addEventListener("click", (e) => {
      if (e.target.closest("[data-ann-del]")) return;
      datasetViewerState.selectedAnnIdx = Number(el.getAttribute("data-ann-idx"));
      renderDatasetViewerOverlay();
      renderDatasetViewerLabelList();
    });
  });
  list.querySelectorAll("[data-ann-del]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      dvPushUndo();
      const i = Number(btn.getAttribute("data-ann-del"));
      datasetViewerState.annotations.splice(i, 1);
      datasetViewerState.selectedAnnIdx = -1;
      renderDatasetViewerOverlay();
      renderDatasetViewerLabelList();
    });
  });
}

function deleteSelectedAnnotation() {
  const idx = datasetViewerState.selectedAnnIdx;
  if (idx < 0 || idx >= datasetViewerState.annotations.length) {
    if (datasetViewerState.annotations.length) {
      dvPushUndo();
      datasetViewerState.annotations.pop();
      renderDatasetViewerOverlay();
      renderDatasetViewerLabelList();
    }
    return;
  }
  dvPushUndo();
  datasetViewerState.annotations.splice(idx, 1);
  datasetViewerState.selectedAnnIdx = -1;
  renderDatasetViewerOverlay();
  renderDatasetViewerLabelList();
}

async function datasetViewerStep(delta) {
  if (datasetViewerState.loading) return;
  const globalIdx = dvGlobalIndex() + delta;
  if (globalIdx < 0 || globalIdx >= datasetViewerState.totalFiles) return;
  const newPage = Math.floor(globalIdx / datasetViewerState.filePageSize) + 1;
  const newIndex = globalIdx % datasetViewerState.filePageSize;
  if (newPage !== datasetViewerState.filePage) {
    if (!(await confirmLeaveIfDirty())) return;
    datasetViewerState.filePage = newPage;
    await loadDatasetViewerImages(true, newIndex);
  } else {
    await navigateDatasetViewerImage(newIndex);
  }
}

async function jumpToNextUnlabeled() {
  const item = datasetViewerState.images[datasetViewerState.index];
  if (!item) return;
  try {
    const q = new URLSearchParams({
      split: datasetViewerState.split,
      path: item.path,
      page_size: String(datasetViewerState.filePageSize),
    });
    const data = await api(
      `/api/platform/datasets/${datasetViewerState.datasetId}/images/next-unlabeled?${q}`
    );
    if (!data.found) {
      toast("没有未标注图片了");
      return;
    }
    await navigateToImagePath(data.path, data.page, data.index_in_page);
  } catch (e) {
    toast(e.message, true);
  }
}

function updateLocalItemAfterSave(item, hasLabels) {
  item.has_labels = hasLabels;
  item.annotate_status = hasLabels ? "labeled" : "unlabeled";
  item.annotate_status_label = hasLabels ? "已标注" : "未标注";
  dvInvalidateLabelCache(item.path);
  const st = datasetViewerState.stats;
  if (st.total_count != null) {
    const wasLabeled = item._wasLabeled;
    if (hasLabels && !wasLabeled) {
      st.labeled_count = (st.labeled_count || 0) + 1;
      st.unlabeled_count = Math.max(0, (st.unlabeled_count || 0) - 1);
    } else if (!hasLabels && wasLabeled) {
      st.labeled_count = Math.max(0, (st.labeled_count || 0) - 1);
      st.unlabeled_count = (st.unlabeled_count || 0) + 1;
    }
  }
  item._wasLabeled = hasLabels;
  updateDvProgressUI();
  renderDatasetFileList();
}

async function persistDatasetLabels() {
  const item = datasetViewerState.images[datasetViewerState.index];
  if (!item) return;
  item._wasLabeled = item.has_labels || item.annotate_status === "labeled";
  await api(
    `/api/platform/datasets/${datasetViewerState.datasetId}/labels?path=${encodeURIComponent(item.path)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        annotations: datasetViewerState.annotations,
        class_names: datasetViewerState.classNames,
      }),
    }
  );
  const hasLabels = datasetViewerState.annotations.length > 0;
  updateLocalItemAfterSave(item, hasLabels);
  dvMarkSaved();
}

async function saveDatasetLabels() {
  try {
    dvSetLoading(true);
    await persistDatasetLabels();
    toast("标注已保存");
  } catch (e) {
    toast(e.message, true);
  } finally {
    dvSetLoading(false);
  }
}

async function saveAndNextDatasetLabels() {
  try {
    dvSetLoading(true);
    await persistDatasetLabels();
    toast("已保存");
    const globalIdx = dvGlobalIndex();
    if (globalIdx < datasetViewerState.totalFiles - 1) {
      await datasetViewerStep(1);
    } else {
      toast("已是最后一张");
    }
  } catch (e) {
    toast(e.message, true);
  } finally {
    dvSetLoading(false);
  }
}

async function deleteDatasetLabels() {
  const item = datasetViewerState.images[datasetViewerState.index];
  if (!item || !confirm("确定清除当前图片全部标注？")) return;
  try {
    dvSetLoading(true);
    await api(
      `/api/platform/datasets/${datasetViewerState.datasetId}/labels?path=${encodeURIComponent(item.path)}`,
      { method: "DELETE" }
    );
    datasetViewerState.annotations = [];
    datasetViewerState.selectedAnnIdx = -1;
    updateLocalItemAfterSave(item, false);
    dvMarkSaved();
    renderDatasetViewerOverlay();
    renderDatasetViewerLabelList();
    toast("标注已清除");
  } catch (e) {
    toast(e.message, true);
  } finally {
    dvSetLoading(false);
  }
}

async function reviewCurrentImage(action) {
  const item = datasetViewerState.images[datasetViewerState.index];
  if (!item) return;
  try {
    await api(
      `/api/platform/datasets/${datasetViewerState.datasetId}/images/review?path=${encodeURIComponent(item.path)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      }
    );
    item.annotate_status = action === "approve" ? "approved" : "rejected";
    item.annotate_status_label = action === "approve" ? "已通过" : "已驳回";
    renderDatasetFileList();
    toast(action === "approve" ? "本张已通过" : "本张已驳回");
    await loadDatasetViewerImages(true);
  } catch (e) {
    toast(e.message, true);
  }
}

async function preAnnotateCurrentImage() {
  const item = datasetViewerState.images[datasetViewerState.index];
  if (!item) return;
  const defaultInst = lastStatus?.config?.default_instance_id || "default";
  openModal({
    title: "AI 预标注",
    wide: true,
    confirmText: "开始标注",
    bodyHtml: `
      <p class="hint">使用推理实例对当前图片进行自动检测，结果需人工检查后保存。</p>
      <div class="form-grid">
        <div class="full"><label>推理实例 ID</label><input type="text" id="dvPreAnnotateInst" value="${platformEsc(defaultInst)}" placeholder="default" /></div>
      </div>`,
    onConfirm: async () => {
      const instanceId = document.getElementById("dvPreAnnotateInst")?.value.trim();
      if (!instanceId) throw new Error("请输入推理实例 ID");
      closeModal();
      try {
        dvSetLoading(true);
        const data = await api(
          `/api/platform/datasets/${datasetViewerState.datasetId}/pre-annotate?path=${encodeURIComponent(item.path)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ instance_id: instanceId, save: false }),
          }
        );
        dvPushUndo();
        datasetViewerState.annotations = data.annotations || [];
        if (data.class_names?.length) {
          datasetViewerState.classNames = data.class_names;
          renderDatasetClassList();
        }
        renderDatasetViewerOverlay();
        renderDatasetViewerLabelList();
        toast(`AI 预标注：${data.count || 0} 个目标（请检查后保存）`);
      } catch (e) {
        toast(e.message, true);
      } finally {
        dvSetLoading(false);
      }
    },
  });
}

function openAddClassModal() {
  openModal({
    title: "添加标注类别",
    confirmText: "添加",
    bodyHtml: `
      <div class="form-grid">
        <div class="full"><label>类别名称</label><input type="text" id="dvNewClassName" placeholder="如：人、车、火" /></div>
      </div>`,
    onConfirm: async () => {
      const name = document.getElementById("dvNewClassName")?.value.trim();
      if (!name) throw new Error("请输入类别名称");
      const names = [...(datasetViewerState.classNames.length ? datasetViewerState.classNames : ["class_0"]), name];
      const data = await api(`/api/platform/datasets/${datasetViewerState.datasetId}/classes`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ class_names: names }),
      });
      datasetViewerState.classNames = data.class_names || names;
      renderDatasetClassList();
      closeModal();
      toast("类别已更新");
    },
  });
}

async function addDatasetClass() {
  openAddClassModal();
}

async function submitDatasetReview() {
  if (datasetViewerState.dirty && !confirm("当前图片有未保存修改，仍要提交？")) return;
  if (!confirm("提交后进入待审核状态，确定提交？")) return;
  try {
    await api(`/api/platform/datasets/${datasetViewerState.datasetId}/review/submit`, { method: "POST" });
    toast("已提交审核");
    await loadDatasetViewerImages(true);
    loadPlatformDatasets();
  } catch (e) {
    toast(e.message, true);
  }
}

async function approveDatasetReview() {
  if (!confirm("审核通过后将可用于模型训练，确定通过？")) return;
  try {
    await api(`/api/platform/datasets/${datasetViewerState.datasetId}/review/approve`, { method: "POST" });
    toast("审核已通过，数据集可用于训练");
    await loadDatasetViewerImages(true);
    loadPlatformDatasets();
  } catch (e) {
    toast(e.message, true);
  }
}

async function rejectDatasetReview() {
  openModal({
    title: "驳回数据集审核",
    confirmText: "确认驳回",
    bodyHtml: `
      <p class="hint">驳回后需继续修改标注并重新提交审核。</p>
      <div class="form-grid">
        <div class="full"><label>驳回原因（可选）</label><input type="text" id="dvRejectReason" placeholder="请说明驳回原因..." /></div>
      </div>`,
    onConfirm: async () => {
      const msg = document.getElementById("dvRejectReason")?.value.trim() || "";
      await api(`/api/platform/datasets/${datasetViewerState.datasetId}/review/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });
      closeModal();
      toast("已驳回，请继续修改标注");
      await loadDatasetViewerImages(true);
      loadPlatformDatasets();
    },
  });
}

function initDatasetViewer() {
  bindDatasetFileListEvents();
  document.getElementById("dvClose")?.addEventListener("click", closeDatasetViewer);
  document.getElementById("dvPrev")?.addEventListener("click", () => datasetViewerStep(-1));
  document.getElementById("dvNext")?.addEventListener("click", () => datasetViewerStep(1));
  document.getElementById("dvNextUnlabeled")?.addEventListener("click", jumpToNextUnlabeled);
  document.getElementById("dvUndo")?.addEventListener("click", dvUndo);
  document.getElementById("dvRedo")?.addEventListener("click", dvRedo);
  document.getElementById("dvToolSelect")?.addEventListener("click", () => setDvTool("select"));
  document.getElementById("dvToolRect")?.addEventListener("click", () => setDvTool("rect"));
  document.getElementById("dvZoomIn")?.addEventListener("click", () => dvZoomBy(0.15));
  document.getElementById("dvZoomOut")?.addEventListener("click", () => dvZoomBy(-0.15));
  document.getElementById("dvZoomReset")?.addEventListener("click", dvResetView);
  document.getElementById("dvClassSelect")?.addEventListener("change", (e) => {
    datasetViewerState.selectedClassId = Number(e.target.value) || 0;
    renderDatasetClassList();
    setDvTool("rect");
  });
  document.getElementById("dvSaveLabels")?.addEventListener("click", saveDatasetLabels);
  document.getElementById("dvSaveNext")?.addEventListener("click", saveAndNextDatasetLabels);
  document.getElementById("dvDeleteLabels")?.addEventListener("click", deleteDatasetLabels);
  document.getElementById("dvPreAnnotate")?.addEventListener("click", preAnnotateCurrentImage);
  document.getElementById("dvAddClass")?.addEventListener("click", addDatasetClass);
  document.getElementById("dvSubmitReview")?.addEventListener("click", submitDatasetReview);
  document.getElementById("dvApproveReview")?.addEventListener("click", approveDatasetReview);
  document.getElementById("dvRejectReview")?.addEventListener("click", rejectDatasetReview);
  document.getElementById("dvImageApprove")?.addEventListener("click", () => reviewCurrentImage("approve"));
  document.getElementById("dvImageReject")?.addEventListener("click", () => reviewCurrentImage("reject"));
  document.getElementById("dvSplitSelect")?.addEventListener("change", async (e) => {
    datasetViewerState.split = e.target.value;
    datasetViewerState.filePage = 1;
    await loadDatasetViewerImages(false, 0);
  });
  document.getElementById("dvStatusFilter")?.addEventListener("change", async (e) => {
    datasetViewerState.statusFilter = e.target.value;
    datasetViewerState.filePage = 1;
    await loadDatasetViewerImages(false, 0);
  });
  document.getElementById("dvFileSearch")?.addEventListener("input", (e) => {
    datasetViewerState.fileSearch = e.target.value;
    clearTimeout(datasetViewerState.fileSearchTimer);
    datasetViewerState.fileSearchTimer = setTimeout(async () => {
      datasetViewerState.filePage = 1;
      await loadDatasetViewerImages(false, 0);
    }, 350);
  });

  const viewport = document.getElementById("dvViewport");
  viewport?.addEventListener("wheel", (e) => {
    if (!document.getElementById("datasetViewer")?.hidden) {
      e.preventDefault();
      dvZoomBy(e.deltaY < 0 ? 0.1 : -0.1, e.clientX, e.clientY);
    }
  }, { passive: false });

  const canvas = document.getElementById("dvCanvas");
  if (canvas) {
    canvas.addEventListener("mousedown", (evt) => {
      if (datasetViewerState.loading) return;
      if (evt.button === 1 || (evt.button === 0 && evt.altKey)) {
        datasetViewerState.isPanning = true;
        datasetViewerState.panStart = { x: evt.clientX, y: evt.clientY, panX: datasetViewerState.panX, panY: datasetViewerState.panY };
        evt.preventDefault();
        return;
      }
      const p = dvCanvasPoint(evt);
      if (datasetViewerState.tool === "select") {
        datasetViewerState.selectedAnnIdx = hitTestAnnotation(p.x, p.y);
        renderDatasetViewerOverlay();
        renderDatasetViewerLabelList();
        return;
      }
      datasetViewerState.drawing = { startX: p.x, startY: p.y, curX: p.x, curY: p.y };
    });
    canvas.addEventListener("mousemove", (evt) => {
      if (datasetViewerState.isPanning && datasetViewerState.panStart) {
        const s = datasetViewerState.panStart;
        datasetViewerState.panX = s.panX + (evt.clientX - s.x);
        datasetViewerState.panY = s.panY + (evt.clientY - s.y);
        applyDvTransform();
        return;
      }
      if (!datasetViewerState.drawing) return;
      const p = dvCanvasPoint(evt);
      datasetViewerState.drawing.curX = p.x;
      datasetViewerState.drawing.curY = p.y;
      scheduleDatasetViewerOverlay();
    });
    const finishDraw = () => {
      if (datasetViewerState.isPanning) {
        datasetViewerState.isPanning = false;
        datasetViewerState.panStart = null;
        return;
      }
      if (!datasetViewerState.drawing) return;
      const d = datasetViewerState.drawing;
      const x = Math.min(d.startX, d.curX);
      const y = Math.min(d.startY, d.curY);
      const w = Math.abs(d.curX - d.startX);
      const h = Math.abs(d.curY - d.startY);
      datasetViewerState.drawing = null;
      const m = dvDisplayMetrics();
      if (m && w >= 6 && h >= 6) {
        dvPushUndo();
        const norm = pixelRectToYolo(x, y, w, h, m.displayW, m.displayH);
        const cid = datasetViewerState.selectedClassId;
        datasetViewerState.annotations.push({
          class_id: cid,
          class_name: datasetViewerState.classNames[cid] || `class_${cid}`,
          cx: norm.cx,
          cy: norm.cy,
          w: norm.w,
          h: norm.h,
        });
        datasetViewerState.selectedAnnIdx = datasetViewerState.annotations.length - 1;
      }
      renderDatasetViewerOverlay();
      renderDatasetViewerLabelList();
    };
    canvas.addEventListener("mouseup", finishDraw);
    canvas.addEventListener("mouseleave", finishDraw);
  }

  window.addEventListener("resize", () => {
    if (!document.getElementById("datasetViewer")?.hidden) scheduleDatasetViewerOverlay();
  });

  document.addEventListener("keydown", (e) => {
    const viewer = document.getElementById("datasetViewer");
    if (viewer?.hidden) return;
    if (e.target.matches("input, textarea, select")) return;

    if (e.key === "Escape") closeDatasetViewer();
    else if (e.key === "ArrowLeft") { e.preventDefault(); datasetViewerStep(-1); }
    else if (e.key === "ArrowRight" || (e.key === " " && !e.ctrlKey)) { e.preventDefault(); datasetViewerStep(1); }
    else if (e.key === "s" && e.ctrlKey && e.shiftKey) { e.preventDefault(); saveAndNextDatasetLabels(); }
    else if (e.key === "s" && e.ctrlKey) { e.preventDefault(); saveDatasetLabels(); }
    else if (e.key === "z" && e.ctrlKey && !e.shiftKey) { e.preventDefault(); dvUndo(); }
    else if ((e.key === "y" && e.ctrlKey) || (e.key === "z" && e.ctrlKey && e.shiftKey)) { e.preventDefault(); dvRedo(); }
    else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); deleteSelectedAnnotation(); }
    else if (e.key === "n" || e.key === "N") { e.preventDefault(); jumpToNextUnlabeled(); }
    else if (e.key === "v" || e.key === "V") setDvTool("select");
    else if (e.key === "r" || e.key === "R") setDvTool("rect");
    else if (/^[1-9]$/.test(e.key)) {
      const idx = Number(e.key) - 1;
      if (idx < (datasetViewerState.classNames.length || 1)) {
        datasetViewerState.selectedClassId = idx;
        renderDatasetClassList();
        document.getElementById("dvClassSelect").value = String(idx);
        setDvTool("rect");
      }
    }
  });
}
