const TOKEN_KEY = "rfdetr_token";

function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

if (getToken()) {
  window.location.href = "/";
}

document.addEventListener("DOMContentLoaded", () => {
  const logoEl = document.getElementById("loginLogo");
  if (logoEl && typeof icon === "function") logoEl.innerHTML = icon("logo", "ico");
});

document.getElementById("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errEl = document.getElementById("loginError");
  const btn = document.getElementById("btnLogin");
  errEl.textContent = "";
  btn.disabled = true;
  btn.textContent = "登录中...";

  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: document.getElementById("username").value.trim(),
        password: document.getElementById("password").value,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "登录失败");
    setToken(data.access_token);
    window.location.href = "/";
  } catch (err) {
    errEl.textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "登 录";
  }
});
