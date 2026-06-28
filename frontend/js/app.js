const API = "";
const TOKEN_KEY = "rfdetr_token";

const MENU_META = {
  dashboard: { label: "总览", icon: "dashboard", subtitle: "系统运行状态概览" },
  models: { label: "模型管理", icon: "models", subtitle: "上传与管理 YOLO / RF-DETR 模型" },
  datasets: { label: "数据集", icon: "datasets", subtitle: "上传 ZIP、在线标注与审核" },
  devices: { label: "设备管理", icon: "devices", subtitle: "视频流与图片流设备维护" },
  gpu: { label: "推理实例", icon: "gpu", subtitle: "绑定模型与设备，启动分析" },
  alerts: { label: "报警管理", icon: "alerts", subtitle: "检测报警记录分页查看" },
  infer: { label: "图片推理", icon: "infer", subtitle: "上传或 URL 图片检测" },
  video: { label: "视频流", icon: "video", subtitle: "实时视频流检测（手动调试）" },
  train: { label: "模型训练", icon: "train", subtitle: "选择模型与数据集训练" },
  settings: { label: "全局默认", icon: "settings", subtitle: "全局默认参数与 Webhook 报警配置" },
  api: { label: "API 文档", icon: "api", subtitle: "REST API 接口说明" },
  users: { label: "用户管理", icon: "users", subtitle: "管理系统用户" },
  roles: { label: "角色管理", icon: "roles", subtitle: "管理角色与菜单权限" },
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let currentUser = null;
let allMenus = [];
let allRoles = [];
let selectedFile = null;
let ws = null;
let lastStatus = null;
let selectedTrainGpus = new Set([0]);
let selectedUserRoles = new Set();
let selectedRoleMenus = new Set();
let modalConfirmHandler = null;
let modalBusy = false;
let cachedUsers = [];
let cachedRolesList = [];
let cachedInstances = [];
let cachedInstanceCfg = {};
let cachedGpus = [];
let modalInstanceGpus = new Set([0]);
let selectedInstanceDevices = new Set();

const tablePageState = {
  instance: { page: 1, pageSize: 10 },
  user: { page: 1, pageSize: 10 },
  role: { page: 1, pageSize: 10 },
};

function paginateRows(rows, state) {
  const total = rows.length;
  const pageSize = state.pageSize;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const page = Math.min(Math.max(1, state.page), totalPages);
  state.page = page;
  const start = (page - 1) * pageSize;
  return {
    page,
    pageSize,
    total,
    totalPages,
    slice: rows.slice(start, start + pageSize),
    rangeStart: total === 0 ? 0 : start + 1,
    rangeEnd: Math.min(start + pageSize, total),
  };
}

function renderTablePagination(containerId, pag, stateKey, onPageChange) {
  const el = $(containerId);
  if (!el) return;
  if (pag.total === 0) {
    el.innerHTML = "";
    el.hidden = true;
    return;
  }
  el.hidden = false;
  const { page, totalPages, total, pageSize, rangeStart, rangeEnd } = pag;
  el.innerHTML = `
    <div class="pagination-info">第 ${rangeStart}-${rangeEnd} 条，共 ${total} 条</div>
    <div class="pagination-controls">
      <button type="button" class="btn btn-secondary btn-sm" data-pg="first" ${page <= 1 ? "disabled" : ""}>首页</button>
      <button type="button" class="btn btn-secondary btn-sm" data-pg="prev" ${page <= 1 ? "disabled" : ""}>上一页</button>
      <span class="pagination-pages">${page} / ${totalPages}</span>
      <button type="button" class="btn btn-secondary btn-sm" data-pg="next" ${page >= totalPages ? "disabled" : ""}>下一页</button>
      <button type="button" class="btn btn-secondary btn-sm" data-pg="last" ${page >= totalPages ? "disabled" : ""}>末页</button>
      <select class="pagination-size" aria-label="每页条数">
        <option value="10" ${pageSize === 10 ? "selected" : ""}>10 条/页</option>
        <option value="20" ${pageSize === 20 ? "selected" : ""}>20 条/页</option>
        <option value="50" ${pageSize === 50 ? "selected" : ""}>50 条/页</option>
      </select>
    </div>`;

  el.querySelector("[data-pg=first]")?.addEventListener("click", () => {
    tablePageState[stateKey].page = 1;
    onPageChange();
  });
  el.querySelector("[data-pg=prev]")?.addEventListener("click", () => {
    tablePageState[stateKey].page = Math.max(1, tablePageState[stateKey].page - 1);
    onPageChange();
  });
  el.querySelector("[data-pg=next]")?.addEventListener("click", () => {
    tablePageState[stateKey].page = Math.min(totalPages, tablePageState[stateKey].page + 1);
    onPageChange();
  });
  el.querySelector("[data-pg=last]")?.addEventListener("click", () => {
    tablePageState[stateKey].page = totalPages;
    onPageChange();
  });
  el.querySelector(".pagination-size")?.addEventListener("change", (e) => {
    tablePageState[stateKey].pageSize = Number(e.target.value);
    tablePageState[stateKey].page = 1;
    onPageChange();
  });
}

function esc(str) {
  if (str == null) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function setModalBusy(busy, loadingText = "处理中...") {
  modalBusy = busy;
  const confirm = $("#modalConfirm");
  const cancel = $("#modalCancel");
  const close = $("#modalClose");
  if (busy) {
    if (confirm) {
      if (!confirm.dataset.defaultText) confirm.dataset.defaultText = confirm.textContent;
      confirm.textContent = loadingText;
      confirm.disabled = true;
    }
    if (cancel) cancel.disabled = true;
    if (close) close.disabled = true;
    $("#modal")?.classList.add("modal-busy");
  } else {
    if (confirm) {
      confirm.disabled = false;
      if (confirm.dataset.defaultText) confirm.textContent = confirm.dataset.defaultText;
    }
    if (cancel) cancel.disabled = false;
    if (close) close.disabled = false;
    $("#modal")?.classList.remove("modal-busy");
  }
}

function openModal({ title, bodyHtml, confirmText = "保存", wide = false, onConfirm }) {
  setModalBusy(false);
  $("#modalTitle").textContent = title;
  $("#modalBody").innerHTML = bodyHtml;
  const confirm = $("#modalConfirm");
  if (confirm) {
    confirm.textContent = confirmText;
    confirm.dataset.defaultText = confirmText;
    confirm.disabled = false;
  }
  $("#modalCancel").disabled = false;
  $("#modalClose").disabled = false;
  $("#modal").classList.toggle("modal-wide", wide);
  modalConfirmHandler = onConfirm;
  $("#modalBackdrop").classList.add("is-open");
  document.body.style.overflow = "hidden";
}

function closeModal() {
  if (modalBusy) return;
  setModalBusy(false);
  $("#modalBackdrop").classList.remove("is-open");
  document.body.style.overflow = "";
  modalConfirmHandler = null;
}

function initModal() {
  $("#modalClose").onclick = closeModal;
  $("#modalCancel").onclick = closeModal;
  $("#modalBackdrop").onclick = (e) => { if (e.target === $("#modalBackdrop")) closeModal(); };
  $("#modalConfirm").onclick = async () => {
    if (!modalConfirmHandler || modalBusy) return;
    try {
      await modalConfirmHandler();
    } catch (e) {
      setModalBusy(false);
      toast(e.message, true);
    }
  };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && $("#modalBackdrop").classList.contains("is-open") && !modalBusy) closeModal();
  });
}

function initGovIcons() {
  const logoEl = $("#brandLogo");
  if (logoEl) logoEl.innerHTML = icon("logo", "ico");
  const tabIcon = $("#activePageTabIcon");
  if (tabIcon) tabIcon.innerHTML = icon("dashboard", "ico ico-btn");
  document.querySelectorAll(".btn-add-icon").forEach((el) => {
    el.innerHTML = icon("add", "ico ico-btn");
  });
}

function initUserDropdown() {
  const dropdown = $("#userDropdown");
  const trigger = $("#userDropdownTrigger");
  const menu = $("#userDropdownMenu");
  if (!dropdown || !trigger || !menu) return;

  trigger.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = !menu.hidden;
    menu.hidden = open;
    trigger.setAttribute("aria-expanded", String(!open));
    dropdown.classList.toggle("is-open", !open);
  });

  document.addEventListener("click", (e) => {
    if (!dropdown.contains(e.target)) {
      menu.hidden = true;
      trigger.setAttribute("aria-expanded", "false");
      dropdown.classList.remove("is-open");
    }
  });
}

function setPreviewImage(previewEl, src) {
  if (!previewEl) return;
  const frame = previewEl.closest(".preview-frame");
  if (!src) {
    previewEl.removeAttribute("src");
    delete previewEl.dataset.fullSrc;
    previewEl.hidden = true;
    frame?.classList.add("is-empty");
    return;
  }
  previewEl.src = src;
  previewEl.dataset.fullSrc = src;
  previewEl.hidden = false;
  frame?.classList.remove("is-empty");
}

function initPreviewFrames() {
  $$(".preview-frame img").forEach((img) => setPreviewImage(img, null));
}

function resolveInstanceDeviceIds(cfg) {
  if (!cfg) return [];
  if (cfg.device_ids?.length) return cfg.device_ids;
  if (cfg.device_id) return [cfg.device_id];
  return [];
}

function instanceDeviceNames(cfg) {
  const ids = resolveInstanceDeviceIds(cfg);
  if (!ids.length) return "—";
  const devices = platformState?.devices || [];
  return ids.map((id) => devices.find((d) => d.id === id)?.name || id).join("、");
}

function buildDeviceChips(container, devices, selectedSet) {
  if (!container) return;
  if (!devices || devices.length === 0) {
    container.innerHTML = '<span class="hint">暂无设备，请先在设备管理中添加</span>';
    return;
  }
  container.innerHTML = devices.map((d) => {
    const checked = selectedSet.has(d.id) ? "checked" : "";
    const cls = selectedSet.has(d.id) ? "gpu-chip selected" : "gpu-chip";
    const typeLabel = d.device_type === "video" ? "视频" : "图片";
    return `<label class="${cls}" data-dev="${d.id}">
      <input type="checkbox" value="${d.id}" ${checked} />
      ${esc(d.name)} · ${typeLabel}
    </label>`;
  }).join("");

  container.querySelectorAll(".gpu-chip").forEach((chip) => {
    chip.addEventListener("click", (e) => {
      if (e.target.tagName === "INPUT") return;
      const cb = chip.querySelector("input");
      cb.checked = !cb.checked;
      cb.dispatchEvent(new Event("change"));
    });
    const cb = chip.querySelector("input");
    cb.addEventListener("change", () => {
      const id = cb.value;
      if (cb.checked) selectedSet.add(id);
      else selectedSet.delete(id);
      buildDeviceChips(container, devices, selectedSet);
    });
  });
}

function matchText(text, keyword) {
  if (!keyword) return true;
  return (text || "").toLowerCase().includes(keyword.toLowerCase());
}

function formatDateTime(iso) {
  if (!iso) return "—";
  try {
    const normalized = String(iso).includes("T") ? iso : String(iso).replace(" ", "T");
    const d = new Date(/[zZ]$/.test(normalized) ? normalized : `${normalized}Z`);
    if (Number.isNaN(d.getTime())) return String(iso).slice(0, 19).replace("T", " ");
    return d.toLocaleString("zh-CN", { hour12: false });
  } catch {
    return String(iso).slice(0, 19).replace("T", " ");
  }
}

function parseAuditDate(iso) {
  if (!iso) return null;
  try {
    const normalized = String(iso).includes("T") ? iso : String(iso).replace(" ", "T");
    const d = new Date(/[zZ]$/.test(normalized) ? normalized : `${normalized}Z`);
    return Number.isNaN(d.getTime()) ? null : d;
  } catch {
    return null;
  }
}

function matchAuditFilters(row, opts = {}) {
  const keyword = (opts.keyword || "").trim().toLowerCase();
  if (keyword) {
    const extra = typeof opts.extraFields === "function"
      ? opts.extraFields(row)
      : (opts.extraFields || []);
    const haystack = [...extra, row.created_by, row.updated_by, row.uploaded_by].filter(Boolean).join(" ").toLowerCase();
    if (!haystack.includes(keyword)) return false;
  }
  const createdBy = opts.createdBy || "";
  if (createdBy && (row.created_by || "") !== createdBy) return false;
  const from = opts.createdFrom ? new Date(`${opts.createdFrom}T00:00:00`) : null;
  const to = opts.createdTo ? new Date(`${opts.createdTo}T23:59:59.999`) : null;
  if (from || to) {
    const createdAt = parseAuditDate(row.created_at);
    if (!createdAt) return false;
    if (from && createdAt < from) return false;
    if (to && createdAt > to) return false;
  }
  return true;
}

function populateAuditUserSelect(selectEl, rows) {
  if (!selectEl) return;
  const prev = selectEl.value;
  const users = [...new Set(rows.flatMap((r) => [r.created_by, r.updated_by, r.uploaded_by].filter(Boolean)))].sort();
  selectEl.innerHTML = `<option value="">全部创建人</option>${users.map((u) => `<option value="${esc(u)}">${esc(u)}</option>`).join("")}`;
  selectEl.value = prev && [...selectEl.options].some((o) => o.value === prev) ? prev : "";
}

function renderAuditCells(row, escFn = esc) {
  return `<td class="cell-muted">${formatDateTime(row.created_at)}</td>
    <td class="cell-muted">${formatDateTime(row.updated_at)}</td>
    <td>${escFn(row.created_by) || "—"}</td>
    <td>${escFn(row.updated_by) || "—"}</td>`;
}

function getAuditFilterOpts(prefix) {
  const el = (id) => document.getElementById(id);
  return {
    keyword: el(`${prefix}Search`)?.value.trim() || "",
    createdBy: el(`${prefix}FilterCreator`)?.value || "",
    createdFrom: el(`${prefix}CreatedFrom`)?.value || "",
    createdTo: el(`${prefix}CreatedTo`)?.value || "",
  };
}

function bindAuditFilterEvents(prefix, stateKey, onPageChange) {
  [`${prefix}Search`, `${prefix}FilterCreator`, `${prefix}CreatedFrom`, `${prefix}CreatedTo`, `${prefix}FilterState`].forEach((id) => {
    const node = document.getElementById(id);
    if (!node) return;
    const evt = node.type === "text" ? "input" : "change";
    node.addEventListener(evt, () => {
      if (stateKey && tablePageState[stateKey]) tablePageState[stateKey].page = 1;
      if (stateKey && typeof platformPageState !== "undefined" && platformPageState[stateKey]) platformPageState[stateKey].page = 1;
      onPageChange();
    });
  });
}

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

function redirectLogin() {
  clearToken();
  window.location.href = "/login";
}

function authHeaders(extra = {}) {
  const token = getToken();
  return token
    ? { ...extra, Authorization: `Bearer ${token}` }
    : { ...extra };
}

/** Append auth token for img/stream URLs (use & when query string already exists). */
function authUrl(url) {
  const token = getToken();
  if (!token || !url) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(token)}`;
}

function toast(msg, isError = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.style.borderColor = isError ? "var(--danger)" : "var(--border)";
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2800);
}

async function api(path, options = {}) {
  const headers = authHeaders(options.headers || {});
  const res = await fetch(`${API}${path}`, { ...options, headers });
  if (res.status === 401) {
    redirectLogin();
    throw new Error("未登录或会话已过期");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.message || res.statusText);
  return data;
}

function renderSidebar(menus) {
  const nav = $("#sidebarNav");
  if (!nav) return;
  const allowed = new Set(menus || []);
  nav.innerHTML = Object.entries(MENU_META)
    .filter(([key]) => allowed.has(key))
    .map(([key, meta]) =>
      `<button class="sidebar-item${key === "dashboard" ? " active" : ""}" data-tab="${key}">
        <span class="sidebar-icon">${icon(meta.icon, "ico")}</span>
        <span class="sidebar-label">${meta.label}</span>
      </button>`
    ).join("");

  nav.querySelectorAll(".sidebar-item").forEach((item) => {
    item.addEventListener("click", () => switchTab(item.dataset.tab));
  });
}

function switchTab(tabKey) {
  $$(".sidebar-item").forEach((t) => t.classList.toggle("active", t.dataset.tab === tabKey));
  $$(".panel").forEach((p) => p.classList.remove("active"));
  const panel = $(`#panel-${tabKey}`);
  if (panel) panel.classList.add("active");
  const meta = MENU_META[tabKey];
  if (meta) {
    $("#pageTitle").textContent = meta.label;
    $("#pageSubtitle").textContent = meta.subtitle;
    const tabLabel = $("#activePageTabLabel");
    const tabIcon = $("#activePageTabIcon");
    if (tabLabel) tabLabel.textContent = meta.label;
    if (tabIcon) tabIcon.innerHTML = icon(meta.icon, "ico ico-btn");
  }
  if (tabKey === "video") startMjpeg();
  if (tabKey === "gpu" || tabKey === "dashboard") refreshStatus();
  if (tabKey === "users") loadUsers();
  if (tabKey === "roles") loadRoles();
  if (typeof refreshPlatformData === "function") refreshPlatformData(tabKey);
}

async function initAuth() {
  if (!getToken()) {
    redirectLogin();
    return false;
  }
  try {
    currentUser = await api("/api/auth/me");
    const displayName = currentUser.display_name || currentUser.username;
    const roleText = currentUser.is_superuser
      ? "超级管理员"
      : (currentUser.roles || []).map((r) => r.name).join(", ") || "无角色";
    $("#topBarUserName").textContent = displayName;
    $("#dropdownUserName").textContent = displayName;
    $("#dropdownUserRoles").textContent = roleText;
    const avatar = $("#userAvatar");
    if (avatar) avatar.textContent = (displayName[0] || "U").toUpperCase();
    renderSidebar(currentUser.menus || []);
    Object.keys(MENU_META).forEach((key) => {
      if (!(currentUser.menus || []).includes(key)) {
        $(`#panel-${key}`)?.remove();
      }
    });
    return true;
  } catch (e) {
    redirectLogin();
    return false;
  }
}

function buildRoleChips(container, roles, selectedSet, onChange) {
  if (!container) return;
  container.innerHTML = (roles || []).map((r) => {
    const checked = selectedSet.has(r.id) ? "checked" : "";
    const cls = selectedSet.has(r.id) ? "gpu-chip selected" : "gpu-chip";
    return `<label class="${cls}">
      <input type="checkbox" value="${r.id}" ${checked} />
      ${r.name}
    </label>`;
  }).join("");
  container.querySelectorAll("input").forEach((cb) => {
    cb.addEventListener("change", () => {
      const id = Number(cb.value);
      if (cb.checked) selectedSet.add(id);
      else selectedSet.delete(id);
      buildRoleChips(container, roles, selectedSet, onChange);
      onChange?.();
    });
  });
}

function buildMenuChips(container, menus, selectedSet) {
  if (!container) return;
  container.innerHTML = (menus || []).map((m) => {
    const checked = selectedSet.has(m.key) ? "checked" : "";
    const cls = selectedSet.has(m.key) ? "gpu-chip selected" : "gpu-chip";
    return `<label class="${cls}">
      <input type="checkbox" value="${m.key}" ${checked} />
      ${icon(MENU_META[m.key]?.icon || "api", "ico ico-btn")} ${m.label}
    </label>`;
  }).join("");
  container.querySelectorAll("input").forEach((cb) => {
    cb.addEventListener("change", () => {
      if (cb.checked) selectedSet.add(cb.value);
      else selectedSet.delete(cb.value);
      buildMenuChips(container, menus, selectedSet);
    });
  });
}

async function loadUsers() {
  if (!currentUser?.is_superuser) return;
  const [data, rolesData] = await Promise.all([
    api("/api/admin/users"),
    api("/api/admin/roles"),
  ]);
  cachedUsers = data.users || [];
  allRoles = rolesData.roles || [];
  const roleFilter = $("#userFilterRole");
  if (roleFilter) {
    const prev = roleFilter.value;
    roleFilter.innerHTML = `<option value="">全部角色</option>${allRoles.map((r) =>
      `<option value="${r.id}">${esc(r.name)}</option>`).join("")}`;
    roleFilter.value = prev && [...roleFilter.options].some((o) => o.value === prev) ? prev : "";
  }
  populateAuditUserSelect($("#userFilterCreator"), cachedUsers);
  renderUserTable();
}

function getFilteredUsers() {
  const keyword = $("#userSearch")?.value.trim() || "";
  const status = $("#userFilterStatus")?.value || "";
  const roleId = $("#userFilterRole")?.value || "";
  const auditOpts = getAuditFilterOpts("user");
  return cachedUsers.filter((u) => {
    if (status === "active" && !u.is_active) return false;
    if (status === "inactive" && u.is_active) return false;
    if (roleId && !(u.roles || []).some((r) => String(r.id) === roleId)) return false;
    const roleNames = (u.roles || []).map((r) => r.name).join(" ");
    return matchAuditFilters(u, {
      keyword,
      createdBy: auditOpts.createdBy,
      createdFrom: auditOpts.createdFrom,
      createdTo: auditOpts.createdTo,
      extraFields: [u.username, u.display_name, roleNames],
    });
  });
}

function renderUserTable() {
  const allRows = getFilteredUsers();
  const pag = paginateRows(allRows, tablePageState.user);
  const rows = pag.slice;
  const tbody = $("#userTableBody");
  const empty = $("#userTableEmpty");
  $("#userTableCount").textContent = `共 ${pag.total} 条`;
  if (!tbody) return;
  if (pag.total === 0) {
    tbody.innerHTML = "";
    if (empty) empty.hidden = false;
    renderTablePagination("#userPagination", pag, "user", renderUserTable);
    return;
  }
  if (empty) empty.hidden = true;
  tbody.innerHTML = rows.map((u) => {
    const roles = u.is_superuser
      ? '<span class="tag tag-default">超级管理员</span>'
      : (u.roles || []).map((r) => `<span class="tag">${esc(r.name)}</span>`).join("") || '<span class="cell-muted">无</span>';
    const statusBadge = `<span class="badge badge-${u.is_active ? "ready" : "stopped"}">${u.is_active ? "启用" : "禁用"}</span>`;
    const actions = u.is_superuser ? "" : `
      <div class="action-group">
        <button class="btn btn-secondary btn-sm btn-icon-text" data-edit-user="${u.id}">${icon("edit", "ico ico-btn")}编辑</button>
        <button class="btn btn-danger btn-sm btn-icon-text" data-del-user="${u.id}">${icon("delete", "ico ico-btn")}删除</button>
      </div>`;
    return `<tr>
      <td><strong>${esc(u.username)}</strong></td>
      <td>${esc(u.display_name || u.username)}</td>
      <td><div class="tag-list">${roles}</div></td>
      <td>${statusBadge}</td>
      ${renderAuditCells(u)}
      <td class="col-actions">${actions}</td>
    </tr>`;
  }).join("");

  tbody.querySelectorAll("[data-edit-user]").forEach((btn) => {
    btn.onclick = () => openUserModal(Number(btn.dataset.editUser));
  });
  tbody.querySelectorAll("[data-del-user]").forEach((btn) => {
    btn.onclick = async () => {
      if (!confirm("确定删除该用户？")) return;
      try {
        await api(`/api/admin/users/${btn.dataset.delUser}`, { method: "DELETE" });
        toast("用户已删除");
        loadUsers();
      } catch (e) { toast(e.message, true); }
    };
  });
  renderTablePagination("#userPagination", pag, "user", renderUserTable);
}

function openUserModal(userId = null) {
  const u = userId ? cachedUsers.find((x) => x.id === userId) : null;
  const isEdit = !!u;
  selectedUserRoles = new Set((u?.roles || []).map((r) => r.id));
  openModal({
    title: isEdit ? "编辑用户" : "新建用户",
    bodyHtml: `
      <div class="form-grid">
        <div><label>用户名</label><input type="text" id="modalUserUsername" ${isEdit ? "disabled" : ""} value="${esc(u?.username || "")}" placeholder="登录用户名" /></div>
        <div><label>显示名称</label><input type="text" id="modalUserDisplayName" value="${esc(u?.display_name || "")}" placeholder="显示名称" /></div>
        <div class="full"><label>密码${isEdit ? "（留空不修改）" : ""}</label><input type="password" id="modalUserPassword" placeholder="${isEdit ? "留空则不修改" : "至少 6 位"}" /></div>
        <div class="full"><label>分配角色</label><div id="modalUserRoleChips" class="gpu-chips"></div></div>
        <div class="full checkbox-row"><input type="checkbox" id="modalUserActive" ${!isEdit || u.is_active ? "checked" : ""} /><label for="modalUserActive">启用账号</label></div>
      </div>`,
    onConfirm: async () => {
      const body = {
        display_name: $("#modalUserDisplayName").value.trim(),
        is_active: $("#modalUserActive").checked,
        role_ids: [...selectedUserRoles],
      };
      const password = $("#modalUserPassword").value;
      if (isEdit) {
        if (password) body.password = password;
        await api(`/api/admin/users/${userId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        toast("用户已更新");
      } else {
        const username = $("#modalUserUsername").value.trim();
        if (!username) throw new Error("请输入用户名");
        if (!password || password.length < 6) throw new Error("密码至少 6 位");
        await api("/api/admin/users", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password, ...body }),
        });
        toast("用户已创建");
      }
      closeModal();
      loadUsers();
    },
  });
  buildRoleChips($("#modalUserRoleChips"), allRoles, selectedUserRoles);
}

async function loadRoles() {
  if (!currentUser?.is_superuser) return;
  const [rolesData, menusData] = await Promise.all([
    api("/api/admin/roles"),
    api("/api/auth/menus"),
  ]);
  cachedRolesList = rolesData.roles || [];
  allMenus = menusData.menus || [];
  populateAuditUserSelect($("#roleFilterCreator"), cachedRolesList);
  renderRoleTable();
}

function getFilteredRoles() {
  const keyword = $("#roleSearch")?.value.trim() || "";
  const auditOpts = getAuditFilterOpts("role");
  return cachedRolesList.filter((r) => {
    const menuLabels = (r.menu_keys || []).map((k) => MENU_META[k]?.label || k).join(" ");
    return matchAuditFilters(r, {
      keyword,
      createdBy: auditOpts.createdBy,
      createdFrom: auditOpts.createdFrom,
      createdTo: auditOpts.createdTo,
      extraFields: [r.name, r.description, menuLabels],
    });
  });
}

function renderRoleTable() {
  const allRows = getFilteredRoles();
  const pag = paginateRows(allRows, tablePageState.role);
  const rows = pag.slice;
  const tbody = $("#roleTableBody");
  const empty = $("#roleTableEmpty");
  $("#roleTableCount").textContent = `共 ${pag.total} 条`;
  if (!tbody) return;
  if (pag.total === 0) {
    tbody.innerHTML = "";
    if (empty) empty.hidden = false;
    renderTablePagination("#rolePagination", pag, "role", renderRoleTable);
    return;
  }
  if (empty) empty.hidden = true;
  tbody.innerHTML = rows.map((r) => {
    const menus = (r.menu_keys || []).map((k) =>
      `<span class="tag">${esc(MENU_META[k]?.label || k)}</span>`).join("") || '<span class="cell-muted">无</span>';
    const isBuiltin = ["admin", "operator", "viewer"].includes(r.name);
    const actions = r.name === "admin" ? "" : `
      <div class="action-group">
        <button class="btn btn-secondary btn-sm btn-icon-text" data-edit-role="${r.id}">${icon("edit", "ico ico-btn")}编辑${isBuiltin ? "权限" : ""}</button>
        ${!isBuiltin ? `<button class="btn btn-danger btn-sm btn-icon-text" data-del-role="${r.id}">${icon("delete", "ico ico-btn")}删除</button>` : ""}
      </div>`;
    return `<tr>
      <td><strong>${esc(r.name)}</strong>${isBuiltin ? ' <span class="tag tag-default">内置</span>' : ""}</td>
      <td class="cell-muted">${esc(r.description || "—")}</td>
      <td><div class="tag-list">${menus}</div></td>
      ${renderAuditCells(r)}
      <td class="col-actions">${actions}</td>
    </tr>`;
  }).join("");

  tbody.querySelectorAll("[data-edit-role]").forEach((btn) => {
    btn.onclick = () => openRoleModal(Number(btn.dataset.editRole));
  });
  tbody.querySelectorAll("[data-del-role]").forEach((btn) => {
    btn.onclick = async () => {
      if (!confirm("确定删除该角色？")) return;
      try {
        await api(`/api/admin/roles/${btn.dataset.delRole}`, { method: "DELETE" });
        toast("角色已删除");
        loadRoles();
      } catch (e) { toast(e.message, true); }
    };
  });
  renderTablePagination("#rolePagination", pag, "role", renderRoleTable);
}

function openRoleModal(roleId = null) {
  const r = roleId ? cachedRolesList.find((x) => x.id === roleId) : null;
  const isEdit = !!r;
  const isBuiltin = r && ["admin", "operator", "viewer"].includes(r.name);
  selectedRoleMenus = new Set(r?.menu_keys || []);
  openModal({
    title: isEdit ? `编辑角色 · ${r.name}` : "新建角色",
    wide: true,
    bodyHtml: `
      <div class="form-grid">
        <div><label>角色名称</label><input type="text" id="modalRoleName" ${isBuiltin ? "disabled" : ""} value="${esc(r?.name || "")}" placeholder="如 operator" /></div>
        <div><label>描述</label><input type="text" id="modalRoleDescription" value="${esc(r?.description || "")}" placeholder="角色说明" /></div>
        <div class="full"><label>菜单权限</label><div id="modalRoleMenuChips" class="gpu-chips"></div></div>
      </div>`,
    onConfirm: async () => {
      const body = {
        name: $("#modalRoleName").value.trim(),
        description: $("#modalRoleDescription").value.trim(),
        menu_keys: [...selectedRoleMenus],
      };
      if (!isEdit && !body.name) throw new Error("请输入角色名称");
      if (isEdit) {
        await api(`/api/admin/roles/${roleId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        toast("角色已更新");
      } else {
        await api("/api/admin/roles", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        toast("角色已创建");
      }
      closeModal();
      loadRoles();
    },
  });
  buildMenuChips($("#modalRoleMenuChips"), allMenus, selectedRoleMenus);
}

function setBadge(el, text, kind) {
  if (!el) return;
  el.textContent = text;
  el.className = `badge badge-${kind}`;
}

function pct(used, total) {
  if (!total) return 0;
  return Math.min(100, Math.round((used / total) * 100));
}

function renderGpuGrid(container, gpus) {
  if (!container) return;
  if (!gpus || gpus.length === 0) {
    container.innerHTML = '<div class="gpu-empty">未检测到 NVIDIA GPU，训练/推理将使用 CPU 或不可用</div>';
    return;
  }
  container.innerHTML = gpus.map((g) => {
    const memPct = pct(g.memory_used_mb, g.memory_total_mb);
    const utilPct = Math.round(g.utilization_gpu || 0);
    const memClass = memPct > 85 ? "warn" : "";
    const utilClass = utilPct > 90 ? "warn" : "";
    return `
      <div class="gpu-card">
        <div class="gpu-card-head">
          <strong title="${g.name}">${g.name.length > 28 ? g.name.slice(0, 28) + "…" : g.name}</strong>
          <span class="gpu-index">GPU ${g.index}</span>
        </div>
        <div class="metric-row">
          <div class="metric-label"><span>显存</span><span>${g.memory_used_mb.toFixed(0)} / ${g.memory_total_mb.toFixed(0)} MB (${memPct}%)</span></div>
          <div class="progress-bar"><div class="progress-fill ${memClass}" style="width:${memPct}%"></div></div>
        </div>
        <div class="metric-row">
          <div class="metric-label"><span>GPU 利用率</span><span>${utilPct}%</span></div>
          <div class="progress-bar"><div class="progress-fill ${utilClass}" style="width:${utilPct}%"></div></div>
        </div>
        <div class="metric-label"><span>PyTorch 已分配</span><span>${g.torch_allocated_mb.toFixed(0)} MB</span></div>
        <div class="metric-label"><span>温度</span><span>${g.temperature_c ? g.temperature_c + " °C" : "N/A"}</span></div>
      </div>`;
  }).join("");
}

function buildGpuChips(container, gpus, selectedSet, onChange) {
  if (!container) return;
  if (!gpus || gpus.length === 0) {
    container.innerHTML = '<span class="hint">无可用 GPU</span>';
    return;
  }
  container.innerHTML = gpus.map((g) => {
    const checked = selectedSet.has(g.index) ? "checked" : "";
    const cls = selectedSet.has(g.index) ? "gpu-chip selected" : "gpu-chip";
    return `<label class="${cls}" data-gpu="${g.index}">
      <input type="checkbox" value="${g.index}" ${checked} />
      GPU ${g.index} · ${g.memory_free_mb.toFixed(0)} MB 空闲
    </label>`;
  }).join("");

  container.querySelectorAll(".gpu-chip").forEach((chip) => {
    chip.addEventListener("click", (e) => {
      if (e.target.tagName === "INPUT") return;
      const cb = chip.querySelector("input");
      cb.checked = !cb.checked;
      cb.dispatchEvent(new Event("change"));
    });
    const cb = chip.querySelector("input");
    cb.addEventListener("change", () => {
      const idx = Number(cb.value);
      if (cb.checked) selectedSet.add(idx);
      else selectedSet.delete(idx);
      if (selectedSet.size === 0) {
        selectedSet.add(idx);
        cb.checked = true;
      }
      buildGpuChips(container, gpus, selectedSet, onChange);
      onChange?.(selectedSet);
    });
  });
}

function instanceStateBadge(state) {
  const map = { ready: "ready", running: "ready", loading: "running", error: "error", stopped: "stopped" };
  return map[state] || "stopped";
}

function renderInstanceCard(inst, cfgItem) {
  const isDefault = cfgItem && lastStatus?.config?.default_instance_id === inst.id;
  const runningCls = inst.state === "ready" ? "instance-card running" : "instance-card";
  return `<div class="${runningCls}">
    <div class="instance-head">
      <div>
        <div class="instance-title">${esc(cfgItem?.name || inst.name)}${isDefault ? " ★" : ""}</div>
        <span class="badge badge-${instanceStateBadge(inst.state)}">${inst.state}</span>
      </div>
    </div>
    <div class="instance-meta">
      <span>GPU<strong>${(inst.gpu_ids || []).join(", ") || "-"}</strong></span>
      <span>设备<strong>${esc(instanceDeviceNames(cfgItem) || "-")}</strong></span>
      <span>权重<strong>${inst.checkpoint ? esc(inst.checkpoint.split(/[/\\]/).pop()) : "预训练"}</strong></span>
    </div>
    <p class="hint">${esc(inst.message || "")}</p>
  </div>`;
}

function getFilteredInstances() {
  const state = $("#instanceFilterState")?.value || "";
  const auditOpts = getAuditFilterOpts("instance");
  return cachedInstances.filter((inst) => {
    if (state && inst.state !== state) return false;
    const cfg = cachedInstanceCfg[inst.id] || {};
    const ckpt = inst.checkpoint || cfg.checkpoint || "";
    const ckptName = ckpt ? ckpt.split(/[/\\]/).pop() : "预训练";
    const devNames = instanceDeviceNames(cfg);
    const modelName = (platformState?.models || []).find((m) => m.id === cfg.model_id)?.name || ckptName;
    return matchAuditFilters(cfg, {
      keyword: auditOpts.keyword,
      createdBy: auditOpts.createdBy,
      createdFrom: auditOpts.createdFrom,
      createdTo: auditOpts.createdTo,
      extraFields: [cfg.name, inst.name, inst.id, ckpt, inst.device, devNames, modelName, inst.state],
    });
  });
}

function renderInstanceTable() {
  const allRows = getFilteredInstances();
  const pag = paginateRows(allRows, tablePageState.instance);
  const rows = pag.slice;
  const tbody = $("#instanceTableBody");
  const empty = $("#instanceTableEmpty");
  const countEl = $("#instanceTableCount");
  if (countEl) countEl.textContent = `共 ${pag.total} 条`;
  if (!tbody) return;
  if (pag.total === 0) {
    tbody.innerHTML = "";
    if (empty) empty.hidden = false;
    renderTablePagination("#instancePagination", pag, "instance", renderInstanceTable);
    return;
  }
  if (empty) empty.hidden = true;
  const defaultId = lastStatus?.config?.default_instance_id;
  tbody.innerHTML = rows.map((inst) => {
    const cfg = cachedInstanceCfg[inst.id] || {};
    const isDefault = inst.id === defaultId;
    const ckpt = inst.checkpoint || cfg.checkpoint || "";
    const ckptName = ckpt ? ckpt.split(/[/\\]/).pop() : "预训练";
    const running = inst.state === "ready";
    const devName = instanceDeviceNames(cfg);
    const devIds = resolveInstanceDeviceIds(cfg);
    const modelName = (platformState?.models || []).find((m) => m.id === cfg.model_id)?.name || (ckpt ? ckptName : "预训练");
    const analysisOn = (platformState?.analysisStatus || []).some((t) => t.instance_id === inst.id && t.running);
    return `<tr class="${running ? "row-running" : ""}">
      <td><strong>${esc(cfg.name || inst.name)}</strong>${isDefault ? ' <span class="tag tag-default">默认</span>' : ""}</td>
      <td class="cell-muted">${esc(inst.id)}</td>
      <td><span class="badge badge-${instanceStateBadge(inst.state)}">${inst.state}</span></td>
      <td>${(inst.gpu_ids || []).join(", ") || "-"}</td>
      <td class="cell-ellipsis" title="${esc(modelName)}">${esc(modelName)}</td>
      <td class="cell-muted">${esc(devName)}</td>
      <td>${cfg.confidence ?? inst.confidence ?? "-"}</td>
      <td class="cell-muted">${inst.last_inference_ms ? inst.last_inference_ms.toFixed(1) + " ms" : "—"}</td>
      ${renderAuditCells(cfg)}
      <td class="col-actions">
        <div class="action-group">
          ${running
            ? `<button class="btn btn-danger btn-sm btn-icon-text" data-inst-action="stop" data-id="${inst.id}">停止</button>`
            : `<button class="btn btn-primary btn-sm btn-icon-text" data-inst-action="start" data-id="${inst.id}">启动</button>`}
          ${running && devIds.length
            ? (analysisOn
              ? `<button class="btn btn-warning btn-sm" data-inst-action="analysis-stop" data-id="${inst.id}">停分析</button>`
              : `<button class="btn btn-primary btn-sm" data-inst-action="analysis-start" data-id="${inst.id}">启分析</button>`)
            : ""}
          <button class="btn btn-secondary btn-sm btn-icon-text" data-inst-action="edit" data-id="${inst.id}" ${running ? "disabled title=\"请先停止实例\"" : ""}>编辑</button>
          ${!isDefault ? `<button class="btn btn-danger btn-sm btn-icon-text" data-inst-action="delete" data-id="${inst.id}" ${running ? "disabled" : ""}>删除</button>` : ""}
        </div>
      </td>
    </tr>`;
  }).join("");
  bindInstanceTableEvents();
  renderTablePagination("#instancePagination", pag, "instance", renderInstanceTable);
}

function bindInstanceTableEvents() {
  $("#instanceTableBody")?.querySelectorAll("[data-inst-action]").forEach((btn) => {
    btn.onclick = async () => {
      const id = btn.dataset.id;
      const action = btn.dataset.instAction;
      try {
        if (action === "start") {
          toast(`正在启动实例 ${id}...`);
          const r = await api(`/api/instances/${id}/start`, { method: "POST" });
          toast(r.message);
        } else if (action === "stop") {
          const r = await api(`/api/instances/${id}/stop`, { method: "POST" });
          toast(r.message);
        } else if (action === "delete") {
          if (!confirm(`确定删除实例 ${id}？`)) return;
          await api(`/api/instances/${id}`, { method: "DELETE" });
          toast("已删除");
        } else if (action === "edit") {
          openInstanceModal(id);
          return;
        } else if (action === "analysis-start") {
          const r = await api(`/api/instances/${id}/analysis/start`, { method: "POST" });
          toast(r.message);
          loadAnalysisStatus?.();
        } else if (action === "analysis-stop") {
          const r = await api(`/api/instances/${id}/analysis/stop`, { method: "POST" });
          toast(r.message);
          loadAnalysisStatus?.();
        }
        refreshStatus();
      } catch (e) { toast(e.message, true); }
    };
  });
}

function openInstanceModal(instanceId = null) {
  const inst = instanceId ? cachedInstances.find((x) => x.id === instanceId) : null;
  const cfg = instanceId ? (cachedInstanceCfg[instanceId] || {}) : {};
  const isEdit = !!inst;
  const defaults = lastStatus?.config?.model || {};
  modalInstanceGpus = new Set(isEdit ? (cfg.gpu_ids || inst.gpu_ids || [0]) : (defaults.gpu_ids || [0]));
  selectedInstanceDevices = new Set(isEdit ? resolveInstanceDeviceIds(cfg) : []);
  openModal({
    title: isEdit ? "编辑推理实例" : "新建推理实例",
    wide: true,
    confirmText: isEdit ? "保存" : "创建",
    bodyHtml: `
      <div class="form-grid">
        <div><label>实例名称</label><input type="text" id="modalInstName" value="${esc(cfg.name || inst?.name || "")}" placeholder="推理实例" /></div>
        <div class="full"><label>选择模型（必填，来自模型管理上传）</label><select id="modalInstModel"><option value="">— 选择模型 —</option></select></div>
        <div class="full"><label>绑定设备（多选，轮询拉取图片推理）</label><div id="modalInstDeviceChips" class="gpu-chips"></div></div>
        <div><label>置信度（报警阈值）</label><input type="number" step="0.01" id="modalInstConfidence" value="${cfg.confidence ?? inst?.confidence ?? defaults.confidence ?? 0.5}" /></div>
        <div><label>分辨率</label><input type="number" step="32" id="modalInstResolution" value="${cfg.resolution ?? inst?.resolution ?? defaults.resolution ?? 576}" /></div>
        <div class="full"><label>类别名（逗号分隔）</label><input type="text" id="modalInstClassNames" value="${esc((cfg.class_names || inst?.class_names || defaults.class_names || []).join(","))}" placeholder="Fire,Smoke" /></div>
        <div class="full"><label>绑定 GPU</label><div id="modalInstGpuChips" class="gpu-chips"></div></div>
      </div>`,
    onConfirm: async () => {
      const modelId = $("#modalInstModel").value;
      if (!modelId) throw new Error("请选择已上传的模型");
      const names = ($("#modalInstClassNames").value || "").split(",").map((s) => s.trim()).filter(Boolean);
      const deviceIds = [...selectedInstanceDevices];
      const body = {
        name: $("#modalInstName").value.trim() || "推理实例",
        model_id: modelId,
        device_ids: deviceIds,
        device_id: deviceIds[0] || "",
        confidence: Number($("#modalInstConfidence").value),
        resolution: Number($("#modalInstResolution").value),
        class_names: names,
        gpu_ids: [...modalInstanceGpus].sort((a, b) => a - b),
      };
      if (isEdit) {
        await api(`/api/instances/${instanceId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        toast("实例配置已保存");
      } else {
        await api("/api/instances", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        toast("实例已创建");
      }
      closeModal();
      refreshStatus();
    },
  });
  refreshModelSelects?.();
  const modelSel = $("#modalInstModel");
  if (modelSel && (cfg.model_id || inst?.model_id)) modelSel.value = cfg.model_id || inst.model_id;
  buildDeviceChips($("#modalInstDeviceChips"), platformState?.devices || [], selectedInstanceDevices);
  buildGpuChips($("#modalInstGpuChips"), cachedGpus, modalInstanceGpus);
}

function renderDetections(container, detections) {
  container.innerHTML = "";
  if (!detections || detections.length === 0) {
    container.innerHTML = '<p class="hint">未检测到目标</p>';
    return;
  }
  detections.forEach((d) => {
    const div = document.createElement("div");
    div.className = "det-item";
    div.innerHTML = `<span><strong>${d.class_name}</strong> #${d.class_id}</span><span>${(d.confidence * 100).toFixed(1)}%</span><span style="grid-column:1/-1;color:var(--muted)">bbox: ${d.bbox.map((v) => v.toFixed(0)).join(", ")}</span>`;
    container.appendChild(div);
  });
}

function showResult(previewEl, metaEl, detEl, result) {
  if (previewEl) {
    setPreviewImage(previewEl, result.image_base64 || null);
  }
  if (metaEl) {
    metaEl.textContent = `实例: ${result.instance_id || "-"} | 来源: ${result.source || "-"} | 检测 ${result.count} 个 | 耗时 ${result.inference_ms.toFixed(1)} ms`;
  }
  renderDetections(detEl, result.detections);
}

function openLightbox(src) {
  if (!src) return;
  $("#lightboxImg").src = src;
  $("#lightbox").classList.add("is-open");
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  $("#lightbox").classList.remove("is-open");
  $("#lightboxImg").src = "";
  document.body.style.overflow = "";
}

function initLightbox() {
  $("#lightboxClose")?.addEventListener("click", (e) => { e.stopPropagation(); closeLightbox(); });
  $("#lightbox")?.addEventListener("click", (e) => { if (e.target === $("#lightbox")) closeLightbox(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeLightbox(); });
  document.body.addEventListener("click", (e) => {
    const frame = e.target.closest(".preview-frame");
    if (!frame || frame.classList.contains("is-empty")) return;
    const img = frame.querySelector("img");
    const src = img?.dataset.fullSrc || img?.currentSrc || img?.src;
    if (!src || src === window.location.href) return;
    e.preventDefault();
    openLightbox(src);
  });
}

function fillInstanceSelects(cfg, instances) {
  const opts = (instances || []).map((inst) => {
    const cfgItem = (cfg.inference_instances || []).find((i) => i.id === inst.id);
    const label = `${cfgItem?.name || inst.name} [${inst.state}] GPU:${(inst.gpu_ids || []).join(",")}`;
    const sel = inst.id === cfg.default_instance_id ? " ★" : "";
    return `<option value="${inst.id}">${label}${sel}</option>`;
  }).join("");

  ["inferInstanceSelect", "inferUrlInstanceSelect", "videoInstanceSelect"].forEach((id) => {
    const el = $(`#${id}`);
    if (!el) return;
    const prev = el.value;
    el.innerHTML = opts;
    if (id === "videoInstanceSelect") el.value = cfg.video?.instance_id || cfg.default_instance_id;
    else el.value = prev && [...el.options].some((o) => o.value === prev) ? prev : cfg.default_instance_id;
  });
}

function updateTrainGpuHint() {
  const n = selectedTrainGpus.size;
  const hint = $("#trainGpuHint");
  if (hint) {
    hint.textContent = n <= 1
      ? `已选 ${n} 张卡：单卡训练（物理 GPU ${[...selectedTrainGpus].join(", ")}）`
      : `已选 ${n} 张卡：多卡 DDP 训练（物理 GPU ${[...selectedTrainGpus].sort((a,b)=>a-b).join(", ")}）`;
  }
}

async function refreshStatus() {
  try {
    const s = await api("/api/status");
    updateUI(s);
  } catch (e) {
    console.warn(e);
  }
}

function updateUI(s) {
  lastStatus = s;
  const gpus = s.gpus || [];
  const instances = s.instances || [];
  const runningCount = instances.filter((i) => i.state === "ready").length;

  setBadge($("#gpuBadge"), `GPU: ${gpus.length} 张`, gpus.length ? "ready" : "stopped");
  setBadge($("#modelBadge"), `实例: ${runningCount}/${instances.length} 运行`, runningCount ? "ready" : "stopped");
  setBadge($("#videoBadge"), `视频: ${s.video_state}`, s.video_state === "running" ? "running" : s.video_state === "error" ? "error" : "stopped");
  setBadge($("#trainBadge"), `训练: ${s.train_state}`, s.train_state === "running" ? "train" : s.train_state === "failed" ? "error" : "stopped");

  $("#videoMessage").textContent = s.video_message;

  renderGpuGrid($("#dashboardGpuGrid"), gpus);

  const cfgMap = Object.fromEntries((s.config.inference_instances || []).map((i) => [i.id, i]));
  cachedInstances = instances;
  cachedInstanceCfg = cfgMap;
  cachedGpus = gpus;
  populateAuditUserSelect($("#instanceFilterCreator"), Object.values(cfgMap));
  renderInstanceTable();

  $("#dashboardInstanceList").innerHTML = instances.length
    ? instances.map((inst) => renderInstanceCard(inst, cfgMap[inst.id])).join("")
    : '<p class="hint">暂无推理实例，请前往「推理实例」创建</p>';

  fillInstanceSelects(s.config, instances);
  fillConfigForms(s.config, gpus);
}

function fillConfigForms(cfg, gpus) {
  $("#cfgConfidence").value = cfg.model.confidence;
  $("#cfgResolution").value = cfg.model.resolution;
  $("#cfgClassNames").value = (cfg.model.class_names || []).join(",");
  $("#cfgOptimize").checked = cfg.model.optimize_inference;
  $("#cfgDefaultInstanceId").value = cfg.default_instance_id || "default";

  $("#videoSourceInput").value = cfg.video.source;
  $("#videoFpsInput").value = cfg.video.fps_limit;
  $("#videoSkipInput").value = cfg.video.skip_frames;
}

function connectWS() {
  const token = getToken();
  if (!token) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  if (ws) {
    ws.onclose = null;
    ws.close();
  }
  ws = new WebSocket(`${proto}://${location.host}/ws/events?token=${encodeURIComponent(token)}`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.event === "inference_done") {
      showResult($("#inferPreview"), $("#inferMeta"), $("#inferDetections"), msg.data);
      showResult($("#dashboardPreview"), null, $("#dashboardDetections"), msg.data);
    }
    if (["instance_started", "instance_stopped", "instances_updated", "config_updated", "video_started", "video_stopped", "train_started", "train_stopped", "model_started", "model_stopped"].includes(msg.event)) {
      refreshStatus();
    }
  };
  ws.onclose = (ev) => {
    if (!getToken()) return;
    if (ev.code === 1008 || ev.code === 4401) return;
    setTimeout(connectWS, 3000);
  };
}

function startMjpeg() {
  const token = getToken();
  $("#mjpegStream").src = `/api/video/mjpeg?t=${Date.now()}&token=${encodeURIComponent(token || "")}`;
}

$("#btnRefreshDashboardGpu").onclick = () => refreshStatus();

$("#btnAddInstance").onclick = () => openInstanceModal();

bindAuditFilterEvents("instance", "instance", renderInstanceTable);
$("#instanceFilterState")?.addEventListener("change", () => {
  tablePageState.instance.page = 1;
  renderInstanceTable();
});

$("#btnAddUser")?.addEventListener("click", () => openUserModal());
bindAuditFilterEvents("user", "user", renderUserTable);
$("#userFilterStatus")?.addEventListener("change", () => {
  tablePageState.user.page = 1;
  renderUserTable();
});
$("#userFilterRole")?.addEventListener("change", () => {
  tablePageState.user.page = 1;
  renderUserTable();
});

$("#btnAddRole")?.addEventListener("click", () => openRoleModal());
bindAuditFilterEvents("role", "role", renderRoleTable);

const uploadZone = $("#uploadZone");
const fileInput = $("#fileInput");
uploadZone.onclick = () => fileInput.click();
uploadZone.ondragover = (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); };
uploadZone.ondragleave = () => uploadZone.classList.remove("dragover");
uploadZone.ondrop = (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) {
    selectedFile = e.dataTransfer.files[0];
    uploadZone.querySelector("p").textContent = selectedFile.name;
  }
};
fileInput.onchange = () => {
  selectedFile = fileInput.files[0];
  if (selectedFile) uploadZone.querySelector("p").textContent = selectedFile.name;
};

$("#btnInferUpload").onclick = async () => {
  if (!selectedFile) return toast("请先选择图片", true);
  const fd = new FormData();
  fd.append("file", selectedFile);
  fd.append("instance_id", $("#inferInstanceSelect").value);
  try {
    toast("推理中...");
    const res = await fetch("/api/infer/image", {
      method: "POST",
      headers: authHeaders(),
      body: fd,
    });
    if (res.status === 401) { redirectLogin(); return; }
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    showResult($("#inferPreview"), $("#inferMeta"), $("#inferDetections"), data);
    showResult($("#dashboardPreview"), null, $("#dashboardDetections"), data);
    toast(`检测完成: ${data.count} 个目标`);
  } catch (e) { toast(e.message, true); }
};

$("#btnInferUrl").onclick = async () => {
  const url = $("#urlInput").value.trim();
  if (!url) return toast("请输入图片 URL", true);
  try {
    const data = await api("/api/infer/url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, instance_id: $("#inferUrlInstanceSelect").value }),
    });
    showResult($("#inferPreview"), $("#inferMeta"), $("#inferDetections"), data);
    toast(`检测完成: ${data.count} 个目标`);
  } catch (e) { toast(e.message, true); }
};

async function saveVideoCfg() {
  await api("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video: {
        source: $("#videoSourceInput").value,
        fps_limit: Number($("#videoFpsInput").value),
        skip_frames: Number($("#videoSkipInput").value),
        instance_id: $("#videoInstanceSelect").value,
      },
    }),
  });
}

$("#btnSaveVideoCfg").onclick = async () => {
  try { await saveVideoCfg(); toast("视频配置已保存"); } catch (e) { toast(e.message, true); }
};

$("#btnStartVideo").onclick = async () => {
  try {
    await saveVideoCfg();
    const r = await api("/api/video/start", { method: "POST" });
    toast(r.message);
    startMjpeg();
    refreshStatus();
  } catch (e) { toast(e.message, true); }
};

$("#btnStopVideo").onclick = async () => {
  try {
    const r = await api("/api/video/stop", { method: "POST" });
    toast(r.message);
    $("#mjpegStream").src = "";
    refreshStatus();
  } catch (e) { toast(e.message, true); }
};

$("#btnSaveModelCfg").onclick = async () => {
  const names = $("#cfgClassNames").value.split(",").map((s) => s.trim()).filter(Boolean);
  try {
    await api("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: {
          confidence: Number($("#cfgConfidence").value),
          resolution: Number($("#cfgResolution").value),
          class_names: names,
          optimize_inference: $("#cfgOptimize").checked,
        },
        default_instance_id: $("#cfgDefaultInstanceId").value || "default",
      }),
    });
    toast("全局默认已保存");
    refreshStatus();
  } catch (e) { toast(e.message, true); }
};

setInterval(async () => {
  if ($("#panel-video").classList.contains("active")) {
    try {
      const v = await api("/api/video/latest");
      if (v.result) {
        $("#videoStats").textContent = `FPS: ${(v.fps || 0).toFixed(1)} | 检测数: ${v.result.count} | 耗时: ${v.result.inference_ms?.toFixed(1)} ms`;
        renderDetections($("#videoDetections"), v.result.detections);
      }
    } catch (_) {}
  }
  if ($("#panel-train").classList.contains("active")) {
    loadTrainingJobs?.();
  }
  if ($("#panel-dashboard").classList.contains("active")) {
    try {
      const gpu = await api("/api/gpu");
      renderGpuGrid($("#dashboardGpuGrid"), gpu.gpus);
      setBadge($("#gpuBadge"), `GPU: ${gpu.count} 张`, gpu.count ? "ready" : "stopped");
    } catch (_) {}
  }
}, 2000);

$("#btnLogout").onclick = () => {
  clearToken();
  if (ws) ws.close();
  redirectLogin();
};

(async function bootstrap() {
  const ok = await initAuth();
  if (!ok) return;
  initGovIcons();
  initUserDropdown();
  initModal();
  initLightbox();
  initPreviewFrames();
  initPlatformPanels?.();
  connectWS();
  refreshStatus();
  refreshPlatformData?.("dashboard");
  setInterval(refreshStatus, 8000);
})();
