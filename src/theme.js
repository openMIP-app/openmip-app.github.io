/* ── openMIP universal theme toggle ── */
(function () {
  const KEY = 'openMIP-theme';

  function isLight() {
    return document.documentElement.classList.contains('light');
  }

  function applyTheme(theme) {
    document.documentElement.classList.toggle('light', theme === 'light');
    localStorage.setItem(KEY, theme);

    const btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = theme === 'light' ? '☀️' : '🌙';

    // Let page-specific code react (e.g. the 3Dmol viewer background).
    document.dispatchEvent(
      new CustomEvent('openMIP:themeChange', { detail: { theme } })
    );
  }

  document.addEventListener('DOMContentLoaded', function () {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;

    // Sync button icon with the class already set by the inline head script.
    btn.textContent = isLight() ? '☀️' : '🌙';

    btn.addEventListener('click', function () {
      applyTheme(isLight() ? 'dark' : 'light');
    });
  });
})();
