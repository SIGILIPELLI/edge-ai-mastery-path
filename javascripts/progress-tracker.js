/* Mastery Path — lesson progress tracker (localStorage, no backend, no accounts). */
(function () {
  "use strict";

  function repoKey() {
    var m = location.pathname.match(/^\/([a-z0-9-]+)\//);
    return m ? m[1] : location.hostname;
  }
  var REPO = repoKey();
  var STORAGE_PREFIX = "mp-done:" + REPO + ":";

  function isDone(slug) {
    return localStorage.getItem(STORAGE_PREFIX + slug) === "1";
  }
  function setDone(slug, done) {
    try {
      if (done) localStorage.setItem(STORAGE_PREFIX + slug, "1");
      else localStorage.removeItem(STORAGE_PREFIX + slug);
    } catch (e) { /* storage disabled/blocked — fail silently */ }
  }
  function slugFromHref(href) {
    try {
      var u = new URL(href, location.href);
      var m = u.pathname.match(/\/(level-\d+)\/([^/]+)\/?$/);
      return m ? m[1] + "/" + m[2] : null;
    } catch (e) { return null; }
  }

  function addCheckboxes() {
    // Scope strictly to the left nav tree — NOT the page's own "On this page" TOC,
    // whose in-page #anchor links would otherwise resolve to the current page's
    // own slug and get double-counted as if they were separate lessons.
    var links = document.querySelectorAll('.md-nav--primary .md-nav__link[href]:not([href^="#"])');
    var seen = {};
    links.forEach(function (link) {
      if (link.dataset.mpChecked) return;
      var slug = slugFromHref(link.getAttribute("href") || "");
      if (!slug || seen[slug]) return;
      seen[slug] = true;
      link.dataset.mpChecked = "1";
      link.dataset.mpSlug = slug;
      var box = document.createElement("input");
      box.type = "checkbox";
      box.className = "mp-progress-check";
      box.checked = isDone(slug);
      box.title = "Mark lesson complete";
      box.addEventListener("click", function (e) {
        e.stopPropagation();
      });
      box.addEventListener("change", function () {
        setDone(slug, box.checked);
        updateBars();
      });
      link.parentNode.insertBefore(box, link);
    });
  }

  function updateBars() {
    var groups = {};
    document.querySelectorAll(".md-nav__link[data-mp-slug]").forEach(function (link) {
      var level = link.dataset.mpSlug.split("/")[0];
      groups[level] = groups[level] || { total: 0, done: 0 };
      groups[level].total++;
      if (isDone(link.dataset.mpSlug)) groups[level].done++;
    });
    document.querySelectorAll(".mp-progress-bar").forEach(function (bar) {
      var level = bar.dataset.level;
      var g = groups[level];
      if (!g || !g.total) { bar.style.display = "none"; return; }
      bar.style.display = "";
      var pct = Math.round((g.done / g.total) * 100);
      var fill = bar.querySelector(".mp-progress-fill");
      var label = bar.querySelector(".mp-progress-label");
      if (fill) fill.style.width = pct + "%";
      if (label) label.textContent = g.done + " / " + g.total + " complete";
    });
  }

  function addPageBanner() {
    var m = location.pathname.match(/\/(level-\d+)\/([^/]+)\/?$/);
    if (!m) return;
    var slug = m[1] + "/" + m[2];
    var content = document.querySelector(".md-content__inner");
    if (!content || content.dataset.mpBanner) return;
    content.dataset.mpBanner = "1";
    var banner = document.createElement("label");
    banner.className = "mp-mark-done";
    var box = document.createElement("input");
    box.type = "checkbox";
    box.checked = isDone(slug);
    box.addEventListener("change", function () {
      setDone(slug, box.checked);
      document.querySelectorAll('.md-nav__link[data-mp-slug="' + slug + '"]').forEach(function (l) {
        var cb = l.previousElementSibling;
        if (cb && cb.classList.contains("mp-progress-check")) cb.checked = box.checked;
      });
      updateBars();
    });
    var span = document.createElement("span");
    span.textContent = box.checked ? "Marked complete" : "Mark this lesson complete";
    box.addEventListener("change", function () {
      span.textContent = box.checked ? "Marked complete" : "Mark this lesson complete";
    });
    banner.appendChild(box);
    banner.appendChild(span);
    content.insertBefore(banner, content.firstChild);
  }

  function addLevelProgressBar() {
    var m = location.pathname.match(/\/(level-\d+)\/(index\/?)?$/);
    var content = document.querySelector(".md-content__inner");
    if (!content || content.dataset.mpBar) return;
    var level = m ? m[1] : (location.pathname.match(/\/(level-\d+)\//) || [])[1];
    if (!level) return;
    content.dataset.mpBar = "1";
    var bar = document.createElement("div");
    bar.className = "mp-progress-bar";
    bar.dataset.level = level;
    bar.innerHTML = '<div class="mp-progress-track"><div class="mp-progress-fill"></div></div><span class="mp-progress-label"></span>';
    content.insertBefore(bar, content.firstChild);
  }

  function init() {
    addCheckboxes();
    addPageBanner();
    addLevelProgressBar();
    updateBars();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
  // mkdocs-material uses instant loading (SPA-like nav) — re-run on navigation.
  if (window.document$ && window.document$.subscribe) {
    window.document$.subscribe(init);
  }
})();
