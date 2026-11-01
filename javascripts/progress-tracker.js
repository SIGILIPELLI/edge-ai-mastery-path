/* Mastery Path — lesson progress tracker (localStorage, no backend, no accounts). */
(function () {
  "use strict";

  // document.currentScript is only valid during this synchronous top-level
  // run, so it's captured here and used later to resolve asset paths (e.g.
  // the certificate signature image) correctly from any page depth.
  var SCRIPT_BASE = (function () {
    var cur = document.currentScript;
    if (cur && cur.src) return cur.src.replace(/javascripts\/progress-tracker\.js(\?.*)?$/, "");
    return "";
  })();
  var SIGNATURE_IMG = new Image();
  var signatureReady = new Promise(function (resolve) {
    SIGNATURE_IMG.onload = resolve;
    SIGNATURE_IMG.onerror = resolve;
  });
  SIGNATURE_IMG.src = SCRIPT_BASE + "assets/signature-vinod.png";

  function repoKey() {
    var m = location.pathname.match(/^\/([a-z0-9-]+)\//);
    return m ? m[1] : location.hostname;
  }
  var REPO = repoKey();
  var STORAGE_PREFIX = "mp-done:" + REPO + ":";
  var POINTS_LESSON = 10;
  var POINTS_PROJECT = 50;

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
  function isProjectSlug(slug) {
    return /project|capstone/i.test(slug || "");
  }
  function pointsFor(slug) {
    return isProjectSlug(slug) ? POINTS_PROJECT : POINTS_LESSON;
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
        refreshAll();
      });
      // Wrap just the checkbox + link pair in their own row, rather than
      // flexing the whole <li> — Material's currently-active nav item
      // also carries an unrelated mobile TOC-toggle <label> as a sibling
      // inside the same <li>, which a whole-<li> flex rule would wrongly
      // pull into the same row.
      var row = document.createElement("span");
      row.className = "mp-nav-row";
      link.parentNode.insertBefore(row, link);
      row.appendChild(box);
      row.appendChild(link);
    });
  }

  // Returns { "level-1": {total, done, points, earned}, ... } from the full
  // site nav (present in the DOM on every page, not just the current section).
  function computeLevelStats() {
    var groups = {};
    document.querySelectorAll(".md-nav__link[data-mp-slug]").forEach(function (link) {
      var slug = link.dataset.mpSlug;
      var level = slug.split("/")[0];
      groups[level] = groups[level] || { total: 0, done: 0, points: 0, earned: 0 };
      var pts = pointsFor(slug);
      groups[level].total++;
      groups[level].points += pts;
      if (isDone(slug)) {
        groups[level].done++;
        groups[level].earned += pts;
      }
    });
    return groups;
  }

  function updateBars(stats) {
    document.querySelectorAll(".mp-progress-bar").forEach(function (bar) {
      var level = bar.dataset.level;
      var g = stats[level];
      if (!g || !g.total) { bar.style.display = "none"; return; }
      bar.style.display = "";
      var pct = Math.round((g.done / g.total) * 100);
      var fill = bar.querySelector(".mp-progress-fill");
      var label = bar.querySelector(".mp-progress-label");
      if (fill) fill.style.width = pct + "%";
      if (label) label.textContent = g.done + " / " + g.total + " complete · " + g.earned + " pts";
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

    var span = document.createElement("span");
    function render() {
      span.textContent = box.checked ? "Marked complete" : "Mark this lesson complete";
    }
    render();

    function mark(done) {
      if (box.checked === done) return;
      box.checked = done;
      render();
      setDone(slug, done);
      document.querySelectorAll('.md-nav__link[data-mp-slug="' + slug + '"]').forEach(function (l) {
        var cb = l.previousElementSibling;
        if (cb && cb.classList.contains("mp-progress-check")) cb.checked = done;
      });
      refreshAll();
    }

    box.addEventListener("change", function () {
      mark(box.checked);
    });
    banner.appendChild(box);
    banner.appendChild(span);

    // Placed at the END of the lesson, right before the prev/next footer —
    // by the time a reader sees this, they've already scrolled past
    // everything. Reaching it, or moving on to the next lesson, is itself
    // the signal that the lesson is done, so auto-check it instead of
    // making the reader click anything.
    content.appendChild(banner);

    if (!box.checked && "IntersectionObserver" in window) {
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            mark(true);
            observer.disconnect();
          }
        });
      }, { threshold: 0.6 });
      observer.observe(banner);
    }

    var nextLink = document.querySelector(".md-footer__link--next");
    if (nextLink && !box.checked) {
      nextLink.addEventListener("click", function () {
        mark(true);
      });
    }
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

  // ---------- Topic tiles: turn the plain "## Modules" list into a tile grid ----------
  function findModulesList() {
    var headings = document.querySelectorAll(".md-content h2");
    for (var i = 0; i < headings.length; i++) {
      if (/^modules\b/i.test(headings[i].textContent.trim())) {
        var el = headings[i].nextElementSibling;
        if (el && el.tagName === "OL") return el;
      }
    }
    return null;
  }

  function enhanceTopicTiles() {
    var list = findModulesList();
    if (!list || list.dataset.mpEnhanced) return;
    list.dataset.mpEnhanced = "1";

    var grid = document.createElement("div");
    grid.className = "mp-topic-grid";

    Array.prototype.forEach.call(list.children, function (li, i) {
      var a = li.querySelector("a");
      if (!a) return;
      var slug = slugFromHref(a.getAttribute("href") || "");
      var done = slug ? isDone(slug) : false;
      var project = isProjectSlug(slug);
      var title = a.textContent.replace(/^Project\s*—\s*/i, "");

      // Some module lines carry extra context after the link, e.g.
      // "[Project — X](...) — a complete end-to-end design, from ...".
      // Preserve it as a tile subtitle instead of silently dropping it.
      var liClone = li.cloneNode(true);
      var aInClone = liClone.querySelector("a");
      if (aInClone) aInClone.remove();
      var extra = liClone.textContent.replace(/^[\s—–\-:]+/, "").trim();

      var tile = document.createElement("a");
      tile.href = a.getAttribute("href");
      tile.className = "mp-topic-tile" + (done ? " is-done" : "") + (project ? " is-project" : "");

      var num = document.createElement("span");
      num.className = "mp-topic-num";
      num.textContent = project ? "★" : String(i + 1).padStart(2, "0");
      tile.appendChild(num);

      var body = document.createElement("span");
      body.className = "mp-topic-body";
      var t = document.createElement("span");
      t.className = "mp-topic-title";
      t.textContent = title;
      body.appendChild(t);
      if (extra) {
        var desc = document.createElement("span");
        desc.className = "mp-topic-desc";
        desc.textContent = extra;
        body.appendChild(desc);
      }
      if (project) {
        var tag = document.createElement("span");
        tag.className = "mp-topic-tag";
        tag.textContent = "Project · " + POINTS_PROJECT + " pts";
        body.appendChild(tag);
      }
      tile.appendChild(body);

      var check = document.createElement("span");
      check.className = "mp-topic-check";
      check.textContent = done ? "✓" : "";
      tile.appendChild(check);

      grid.appendChild(tile);
    });

    if (!grid.children.length) return;
    list.parentNode.replaceChild(grid, list);
  }

  // ---------- Level tiles: turn the homepage's "How the program is organized" table into cards ----------
  function findLevelTable() {
    var tables = document.querySelectorAll(".md-content table");
    for (var i = 0; i < tables.length; i++) {
      var firstTh = tables[i].querySelector("thead th");
      if (firstTh && /^level$/i.test(firstTh.textContent.trim())) return tables[i];
    }
    return null;
  }

  function enhanceLevelTable(stats) {
    var table = findLevelTable();
    if (!table || table.dataset.mpEnhanced) return;
    table.dataset.mpEnhanced = "1";

    var rows = table.querySelectorAll("tbody tr");
    var grid = document.createElement("div");
    grid.className = "mp-level-grid";

    rows.forEach(function (row) {
      var cells = row.querySelectorAll("td");
      if (cells.length < 3) return;
      var link = cells[0].querySelector("a");
      if (!link) return;
      var href = link.getAttribute("href");
      var levelMatch = href.match(/(level-\d+)/);
      var level = levelMatch ? levelMatch[1] : null;
      var g = level ? stats[level] : null;

      var card = document.createElement("a");
      card.href = href;
      card.className = "mp-level-card";

      var head = document.createElement("span");
      head.className = "mp-level-name";
      head.textContent = link.textContent;
      card.appendChild(head);

      var focus = document.createElement("span");
      focus.className = "mp-level-focus";
      focus.textContent = cells[1].textContent;
      card.appendChild(focus);

      var modules = document.createElement("span");
      modules.className = "mp-level-modules";
      modules.textContent = cells[2].textContent;
      card.appendChild(modules);

      if (g && g.total) {
        var pct = Math.round((g.done / g.total) * 100);
        var track = document.createElement("span");
        track.className = "mp-level-track";
        var fill = document.createElement("span");
        fill.className = "mp-level-fill";
        fill.style.width = pct + "%";
        track.appendChild(fill);
        card.appendChild(track);

        var status = document.createElement("span");
        status.className = "mp-level-status";
        status.textContent = g.done >= g.total ? "✓ Complete · " + g.earned + " pts" : g.done + " / " + g.total + " complete";
        card.appendChild(status);
      }

      grid.appendChild(card);
    });

    table.parentNode.replaceChild(grid, table);
  }

  // ---------- Points + certificate summary on the homepage hero ----------
  function totalCourseStats(stats) {
    var out = { total: 0, done: 0, points: 0, earned: 0, levels: 0, levelsDone: 0 };
    Object.keys(stats).forEach(function (level) {
      var g = stats[level];
      out.total += g.total;
      out.done += g.done;
      out.points += g.points;
      out.earned += g.earned;
      out.levels++;
      if (g.total && g.done >= g.total) out.levelsDone++;
    });
    return out;
  }

  function addHeroSummary(stats) {
    var hero = document.querySelector(".mp-hero-lockup");
    if (!hero || hero.dataset.mpSummary) return;
    var course = totalCourseStats(stats);
    if (!course.done) return; // nothing to show for a fresh visitor
    hero.dataset.mpSummary = "1";

    var box = document.createElement("div");
    box.className = "mp-hero-summary";
    box.innerHTML =
      '<span class="mp-hero-points">🏆 ' + course.earned.toLocaleString() + ' points</span>' +
      '<span class="mp-hero-sep">·</span>' +
      '<span>' + course.done + ' / ' + course.total + ' lessons complete</span>';
    hero.parentNode.insertBefore(box, hero.nextSibling);
  }

  // ---------- Certificates ----------
  var COURSE_NAME = (function () {
    var m = document.querySelector('meta[property="og:site_name"]');
    return m ? m.content : document.title;
  })();

  function cleanHeadingText(h1) {
    if (!h1) return "";
    var clone = h1.cloneNode(true);
    var badge = clone.querySelector(".level-badge");
    if (badge) badge.remove();
    var link = clone.querySelector(".headerlink");
    if (link) link.remove();
    return clone.textContent.trim();
  }

  function addCertificateButton(container, kind, levelName) {
    if (container.dataset.mpCert) return;
    container.dataset.mpCert = "1";
    var card = document.createElement("div");
    card.className = "mp-cert-cta";
    card.innerHTML =
      '<span class="mp-cert-cta-icon">🎓</span>' +
      '<span class="mp-cert-cta-text">' +
      '<strong>' + (kind === "course" ? "Course complete!" : "Level complete!") + '</strong>' +
      '<span>Claim your ' + (kind === "course" ? "course completion" : "level") + ' certificate.</span>' +
      '</span>' +
      '<button type="button" class="mp-cert-cta-btn">Get certificate</button>';
    card.querySelector("button").addEventListener("click", function () {
      openCertModal(kind, levelName);
    });
    container.appendChild(card);
  }

  function checkCertificates(stats) {
    var course = totalCourseStats(stats);

    // Level-page certificate: only on that level's own overview page.
    var m = location.pathname.match(/\/(level-\d+)\/(index\/?)?$/);
    if (m) {
      var level = m[1];
      var g = stats[level];
      if (g && g.total && g.done >= g.total) {
        var content = document.querySelector(".md-content__inner");
        var h1 = document.querySelector(".md-content h1");
        var levelName = cleanHeadingText(h1) || level;
        if (content) addCertificateButton(content, "level", levelName);
      }
    }

    // Course-wide certificate: on the homepage, once every level is done.
    if (location.pathname.match(/^\/[a-z0-9-]+\/?$/) && course.levels && course.levelsDone >= course.levels) {
      var hero = document.querySelector(".mp-hero-lockup");
      var host = hero ? hero.parentNode : document.querySelector(".md-content__inner");
      if (host) addCertificateButton(host, "course", null);
    }
  }

  var NAME_KEY = "mp-learner-name";

  function openCertModal(kind, levelName) {
    var existing = document.getElementById("mp-cert-modal");
    if (existing) existing.remove();

    var overlay = document.createElement("div");
    overlay.id = "mp-cert-modal";
    overlay.className = "mp-modal-overlay";
    var savedName = "";
    try { savedName = localStorage.getItem(NAME_KEY) || ""; } catch (e) {}

    overlay.innerHTML =
      '<div class="mp-modal" role="dialog" aria-modal="true" aria-label="Claim certificate">' +
      '<button type="button" class="mp-modal-close" aria-label="Close">&times;</button>' +
      '<h3>🎓 Claim your certificate</h3>' +
      '<p>' + (kind === "course"
        ? "You've completed every level of " + escapeHtml(COURSE_NAME) + ". What name should the certificate show?"
        : "You've completed " + escapeHtml(levelName) + ". What name should the certificate show?") + '</p>' +
      '<input type="text" class="mp-modal-input" placeholder="Your full name" maxlength="80" value="' + escapeHtml(savedName) + '">' +
      '<div class="mp-modal-actions">' +
      '<button type="button" class="mp-modal-cancel">Cancel</button>' +
      '<button type="button" class="mp-modal-generate">Generate certificate</button>' +
      '</div>' +
      '</div>';

    document.body.appendChild(overlay);
    var input = overlay.querySelector(".mp-modal-input");
    input.focus();
    input.select();

    function close() { overlay.remove(); }
    overlay.addEventListener("click", function (e) { if (e.target === overlay) close(); });
    overlay.querySelector(".mp-modal-close").addEventListener("click", close);
    overlay.querySelector(".mp-modal-cancel").addEventListener("click", close);
    function submit() {
      var name = input.value.trim();
      if (!name) { input.focus(); return; }
      try { localStorage.setItem(NAME_KEY, name); } catch (e) {}
      signatureReady.then(function () {
        generateCertificate(name, kind, levelName);
      });
      close();
    }
    overlay.querySelector(".mp-modal-generate").addEventListener("click", submit);
    input.addEventListener("keydown", function (e) { if (e.key === "Enter") submit(); });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function generateCertificate(name, kind, levelName) {
    var W = 1600, H = 1040;
    var canvas = document.createElement("canvas");
    canvas.width = W; canvas.height = H;
    var ctx = canvas.getContext("2d");

    // background
    ctx.fillStyle = "#FAF7EF";
    ctx.fillRect(0, 0, W, H);

    // outer border
    ctx.strokeStyle = "#2FA98D";
    ctx.lineWidth = 6;
    ctx.strokeRect(36, 36, W - 72, H - 72);
    ctx.strokeStyle = "#E0B457";
    ctx.lineWidth = 2;
    ctx.strokeRect(56, 56, W - 112, H - 112);

    // brand mark (ascending bars) — small letterhead-style mark up top
    var bx = W / 2 - 50, by = 80;
    var bars = [
      { x: 0, h: 24, c: "#2FA98D" },
      { x: 17, h: 40, c: "#7FC9B7" },
      { x: 34, h: 57, c: "#E0B457" },
      { x: 51, h: 74, c: "#C97B4A" }
    ];
    bars.forEach(function (b) {
      ctx.fillStyle = b.c;
      roundRect(ctx, bx + b.x, by + (74 - b.h), 12, b.h, 4);
      ctx.fill();
    });
    ctx.fillStyle = "#22261f";
    ctx.beginPath();
    ctx.arc(bx + 6, by + 17, 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.textAlign = "center";
    ctx.fillStyle = "#565f52";
    ctx.font = "700 18px Georgia, 'Times New Roman', serif";
    ctx.fillText("M A S T E R Y   P A T H", W / 2, by + 108);

    // big title, matching a classic "CERTIFICATE OF COMPLETION" letterhead
    ctx.fillStyle = "#22261f";
    ctx.font = "700 66px Georgia, 'Times New Roman', serif";
    ctx.fillText("C E R T I F I C A T E", W / 2, 325);

    ctx.fillStyle = "#565f52";
    ctx.font = "700 20px Georgia, 'Times New Roman', serif";
    ctx.fillText("O F   C O M P L E T I O N", W / 2, 362);

    ctx.font = "400 22px Georgia, 'Times New Roman', serif";
    ctx.fillStyle = "#565f52";
    ctx.fillText("This certificate is given to:", W / 2, 420);

    ctx.font = "700 60px Georgia, 'Times New Roman', serif";
    ctx.fillStyle = "#22261f";
    ctx.fillText(name, W / 2, 495);
    ctx.strokeStyle = "#E0B457";
    ctx.lineWidth = 2;
    ctx.beginPath();
    var nameWidth = Math.min(ctx.measureText(name).width + 80, W - 300);
    ctx.moveTo(W / 2 - nameWidth / 2, 518);
    ctx.lineTo(W / 2 + nameWidth / 2, 518);
    ctx.stroke();

    var today = new Date();
    var dateStr = today.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });

    ctx.font = "400 24px Georgia, 'Times New Roman', serif";
    ctx.fillStyle = "#565f52";
    var line2 = (kind === "course"
      ? "has successfully completed the complete " + COURSE_NAME
      : "has successfully completed " + (levelName || "a level") + " of " + COURSE_NAME)
      + " on " + dateStr + ". Their dedication and commitment to the learning"
      + " process are truly commendable.";
    wrapText(ctx, line2, W / 2, 565, W - 340, 34);

    // single centered signature: the real embedded signature image over a
    // name line, rather than the old two-signature left/right layout.
    var sigW = 260, sigH = SIGNATURE_IMG.naturalHeight
      ? Math.round(sigW * (SIGNATURE_IMG.naturalHeight / SIGNATURE_IMG.naturalWidth))
      : 109;
    var sigY = 760;
    if (SIGNATURE_IMG.complete && SIGNATURE_IMG.naturalWidth > 0) {
      ctx.drawImage(SIGNATURE_IMG, W / 2 - sigW / 2, sigY - sigH, sigW, sigH);
    }

    var lineY = sigY + 22, labelY = lineY + 32;
    ctx.strokeStyle = "#565f52";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(W / 2 - 150, lineY);
    ctx.lineTo(W / 2 + 150, lineY);
    ctx.stroke();

    ctx.font = "700 22px Georgia, 'Times New Roman', serif";
    ctx.fillStyle = "#22261f";
    ctx.fillText("VINOD KUMAR S", W / 2, labelY);

    ctx.font = "600 13px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = "#565f52";
    ctx.fillText("F O U N D E R ,   M A S T E R Y   P A T H", W / 2, labelY + 24);

    var certId = "MP-" + REPO.replace("-mastery-path", "").toUpperCase().slice(0, 8) + "-" + Date.now().toString(36).toUpperCase();
    ctx.font = "400 15px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillStyle = "#9aa092";
    ctx.fillText("Certificate ID: " + certId + "  ·  sigilipelli.github.io/" + REPO, W / 2, H - 70);

    var url = canvas.toDataURL("image/png");
    var a = document.createElement("a");
    a.href = url;
    var fileLevel = kind === "course" ? "course" : (levelName || "level").toLowerCase().replace(/[^a-z0-9]+/g, "-");
    a.download = REPO + "-" + fileLevel + "-certificate.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function wrapText(ctx, text, cx, y, maxWidth, lineHeight) {
    var words = text.split(" ");
    var line = "";
    var lines = [];
    for (var i = 0; i < words.length; i++) {
      var test = line + words[i] + " ";
      if (ctx.measureText(test).width > maxWidth && line) {
        lines.push(line.trim());
        line = words[i] + " ";
      } else {
        line = test;
      }
    }
    lines.push(line.trim());
    var startY = y - ((lines.length - 1) * lineHeight) / 2;
    lines.forEach(function (l, i) { ctx.fillText(l, cx, startY + i * lineHeight); });
  }

  // ---------- Orchestration ----------
  function refreshAll() {
    var stats = computeLevelStats();
    updateBars(stats);
    addHeroSummary(stats);
    checkCertificates(stats);
  }

  function init() {
    addCheckboxes();
    addPageBanner();
    addLevelProgressBar();
    var stats = computeLevelStats();
    enhanceTopicTiles();
    enhanceLevelTable(stats);
    updateBars(stats);
    addHeroSummary(stats);
    checkCertificates(stats);
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
