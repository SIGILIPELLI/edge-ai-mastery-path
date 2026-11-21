/* Mastery Path - final course quiz (localStorage only, no backend). Renders #mp-quiz-root from #mp-quiz-data. */
(function () {
  "use strict";
  function repoKey() {
    var m = location.pathname.match(/^\/([a-z0-9-]+)\//);
    return m ? m[1] : location.hostname;
  }
  var REPO = repoKey();
  // Safe rich text: ```code blocks``` and `inline code` become <pre><code>/<code>; everything else is plain text.
  function setRich(el, text) {
    el.textContent = "";
    String(text).split(/(```[\s\S]*?```)/).forEach(function (part) {
      if (/^```[\s\S]*```$/.test(part)) {
        var pre = document.createElement("pre"), code = document.createElement("code");
        code.textContent = part.replace(/^```[a-z]*\n?/, "").replace(/\n?```$/, "");
        pre.className = "mp-quiz-code"; pre.appendChild(code); el.appendChild(pre);
      } else {
        part.split(/(`[^`\n]+`)/).forEach(function (seg) {
          if (/^`[^`\n]+`$/.test(seg)) {
            var c = document.createElement("code"); c.textContent = seg.slice(1, -1); el.appendChild(c);
          } else if (seg) { el.appendChild(document.createTextNode(seg)); }
        });
      }
    });
  }
  // ---------- Final quiz ----------
  var QUIZ_PASS_PCT = 70;
  var QUIZ_KEY = "mp-quiz:" + REPO;

  function quizStatus() {
    try {
      var raw = localStorage.getItem(QUIZ_KEY);
      if (!raw) return { done: false };
      var data = JSON.parse(raw);
      var pct = Math.round((data.score / data.total) * 100);
      return { done: true, passed: pct >= QUIZ_PASS_PCT, score: data.score, total: data.total, pct: pct };
    } catch (e) { return { done: false }; }
  }

  function saveQuizResult(score, total) {
    try {
      localStorage.setItem(QUIZ_KEY, JSON.stringify({ score: score, total: total, timestamp: Date.now() }));
    } catch (e) { /* storage disabled/blocked — fail silently */ }
  }

  function renderQuiz() {
    var root = document.getElementById("mp-quiz-root");
    var dataEl = document.getElementById("mp-quiz-data");
    if (!root || !dataEl || root.dataset.mpQuiz) return;
    root.dataset.mpQuiz = "1";

    var questions;
    try { questions = JSON.parse(dataEl.textContent); } catch (e) { return; }
    if (!questions || !questions.length) return;

    function renderSummary(result) {
      root.innerHTML = "";
      var box = document.createElement("div");
      box.className = "mp-quiz-summary" + (result.passed ? " is-pass" : " is-fail");
      box.innerHTML =
        '<div class="mp-quiz-summary-score">' + result.pct + '%</div>' +
        '<div class="mp-quiz-summary-text">' +
        '<strong>' + (result.passed ? "You passed!" : "Not quite — try again.") + '</strong>' +
        '<span>' + result.score + ' / ' + result.total + ' correct · ' + QUIZ_PASS_PCT + '% needed to pass</span>' +
        '</div>' +
        '<button type="button" class="mp-quiz-retake">Retake quiz</button>';
      box.querySelector(".mp-quiz-retake").addEventListener("click", renderForm);
      root.appendChild(box);
    }

    function renderForm() {
      root.innerHTML = "";
      var form = document.createElement("form");
      form.className = "mp-quiz-form";
      form.setAttribute("novalidate", "");

      questions.forEach(function (q, qi) {
        var fieldset = document.createElement("fieldset");
        fieldset.className = "mp-quiz-question";
        var legend = document.createElement("legend");
        setRich(legend, (qi + 1) + ". " + q.question);
        fieldset.appendChild(legend);

        var optList = document.createElement("div");
        optList.className = "mp-quiz-options";
        q.options.forEach(function (opt, oi) {
          var label = document.createElement("label");
          label.className = "mp-quiz-option";
          var input = document.createElement("input");
          input.type = "radio";
          input.name = "mp-quiz-q" + qi;
          input.value = String(oi);
          label.appendChild(input);
          var span = document.createElement("span");
          setRich(span, opt);
          label.appendChild(span);
          optList.appendChild(label);
        });
        fieldset.appendChild(optList);

        var explain = document.createElement("div");
        explain.className = "mp-quiz-explain";
        explain.hidden = true;
        fieldset.appendChild(explain);

        form.appendChild(fieldset);
      });

      var warning = document.createElement("div");
      warning.className = "mp-quiz-warning";
      warning.hidden = true;
      warning.textContent = "Please answer every question before submitting.";
      form.appendChild(warning);

      var submitBtn = document.createElement("button");
      submitBtn.type = "submit";
      submitBtn.className = "mp-quiz-submit";
      submitBtn.textContent = "Submit quiz";
      form.appendChild(submitBtn);

      form.addEventListener("submit", function (e) {
        e.preventDefault();
        var answers = questions.map(function (q, qi) {
          var checked = form.querySelector('input[name="mp-quiz-q' + qi + '"]:checked');
          return checked ? parseInt(checked.value, 10) : -1;
        });
        if (answers.indexOf(-1) !== -1) {
          warning.hidden = false;
          return;
        }
        warning.hidden = true;

        var score = 0;
        questions.forEach(function (q, qi) {
          var fieldset = form.querySelectorAll(".mp-quiz-question")[qi];
          var options = fieldset.querySelectorAll(".mp-quiz-option");
          var isCorrect = answers[qi] === q.correctIndex;
          if (isCorrect) score++;
          options.forEach(function (optEl, oi) {
            optEl.querySelector("input").disabled = true;
            if (oi === q.correctIndex) optEl.classList.add("is-correct");
            else if (oi === answers[qi]) optEl.classList.add("is-wrong");
          });
          var explain = fieldset.querySelector(".mp-quiz-explain");
          setRich(explain, (isCorrect ? "✓ Correct. " : "✗ ") + q.explanation);
          explain.hidden = false;
        });

        // .mp-quiz-submit sets its own `display`, which as an author rule
        // beats the [hidden] UA style regardless of specificity — an
        // inline style is needed to actually hide it after grading.
        submitBtn.style.display = "none";
        var total = questions.length;
        var pct = Math.round((score / total) * 100);
        saveQuizResult(score, total);

        var resultBanner = document.createElement("div");
        resultBanner.className = "mp-quiz-summary" + (pct >= QUIZ_PASS_PCT ? " is-pass" : " is-fail");
        resultBanner.innerHTML =
          '<div class="mp-quiz-summary-score">' + pct + '%</div>' +
          '<div class="mp-quiz-summary-text">' +
          '<strong>' + (pct >= QUIZ_PASS_PCT ? "You passed!" : "Not quite — try again.") + '</strong>' +
          '<span>' + score + ' / ' + total + ' correct · ' + QUIZ_PASS_PCT + '% needed to pass</span>' +
          '</div>' +
          '<button type="button" class="mp-quiz-retake">Retake quiz</button>';
        resultBanner.querySelector(".mp-quiz-retake").addEventListener("click", renderForm);
        form.insertBefore(resultBanner, form.firstChild);
      });

      root.appendChild(form);
    }

    var existing = quizStatus();
    if (existing.done) renderSummary(existing);
    else renderForm();
  }


  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", renderQuiz); else renderQuiz();
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(renderQuiz);
})();
