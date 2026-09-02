---
title: "Learn Edge AI Free: Beginner to Master Course"
description: "Free Edge AI course from beginner to advanced -- hands-on lessons on running AI on-device, with real projects. Part of a 37-course free learning library."
---

# Edge AI Mastery Path

<div class="mp-hero-lockup">
  <div class="mp-hero-globe">
    <svg viewBox="0 0 160 160" class="mp-globe-svg">
      <defs>
        <clipPath id="mp-globe-clip"><circle cx="80" cy="80" r="74"/></clipPath>
        <radialGradient id="mp-sheen" cx="32%" cy="26%" r="60%">
          <stop offset="0%" stop-color="#ffffff" stop-opacity="0.5"/>
          <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <circle cx="80" cy="80" r="74" fill="#FAF7EF"/>
      <g clip-path="url(#mp-globe-clip)">
        <g class="mp-globe-strip">
          <!-- tile 1: 0-160 -->
          <g stroke="#2FA98D" stroke-width="0.6" opacity="0.28" fill="none">
            <ellipse cx="80" cy="80" rx="18" ry="74"/>
            <ellipse cx="80" cy="80" rx="42" ry="74"/>
            <ellipse cx="80" cy="80" rx="66" ry="74"/>
            <line x1="6" y1="80" x2="154" y2="80"/>
            <line x1="10" y1="50" x2="150" y2="50"/>
            <line x1="10" y1="110" x2="150" y2="110"/>
          </g>
          <path d="M62,18 Q78,14 92,20 Q98,30 92,42 Q84,50 74,46 Q64,40 62,28 Z" fill="#2FA98D"/>
          <path d="M58,42 Q80,36 96,46 Q100,58 90,68 Q76,74 64,66 Q56,56 58,42 Z" fill="#E0B457"/>
          <path d="M70,68 Q80,66 84,74 Q82,84 74,86 Q66,82 68,74 Z" fill="#E0B457"/>
          <circle cx="92" cy="72" r="4" fill="#2FA98D"/>
          <path d="M76,86 Q92,82 102,94 Q108,110 100,126 Q92,138 80,136 Q72,124 74,108 Q72,96 76,86 Z" fill="#E0B457"/>
          <path d="M92,100 Q104,98 108,110 Q106,122 96,124 Q88,118 90,108 Z" fill="#7FC9B7"/>
          <circle cx="128" cy="34" r="7" fill="#C97B4A"/>

          <!-- tile 2: 160-320 (identical, offset for seamless loop) -->
          <g transform="translate(160,0)">
            <g stroke="#2FA98D" stroke-width="0.6" opacity="0.28" fill="none">
              <ellipse cx="80" cy="80" rx="18" ry="74"/>
              <ellipse cx="80" cy="80" rx="42" ry="74"/>
              <ellipse cx="80" cy="80" rx="66" ry="74"/>
              <line x1="6" y1="80" x2="154" y2="80"/>
              <line x1="10" y1="50" x2="150" y2="50"/>
              <line x1="10" y1="110" x2="150" y2="110"/>
            </g>
            <path d="M62,18 Q78,14 92,20 Q98,30 92,42 Q84,50 74,46 Q64,40 62,28 Z" fill="#2FA98D"/>
            <path d="M58,42 Q80,36 96,46 Q100,58 90,68 Q76,74 64,66 Q56,56 58,42 Z" fill="#E0B457"/>
            <path d="M70,68 Q80,66 84,74 Q82,84 74,86 Q66,82 68,74 Z" fill="#E0B457"/>
            <circle cx="92" cy="72" r="4" fill="#2FA98D"/>
            <path d="M76,86 Q92,82 102,94 Q108,110 100,126 Q92,138 80,136 Q72,124 74,108 Q72,96 76,86 Z" fill="#E0B457"/>
            <path d="M92,100 Q104,98 108,110 Q106,122 96,124 Q88,118 90,108 Z" fill="#7FC9B7"/>
            <circle cx="128" cy="34" r="7" fill="#C97B4A"/>
          </g>
        </g>
      </g>
      <circle cx="80" cy="80" r="74" fill="none" stroke="#2FA98D" stroke-width="4"/>
      <circle cx="80" cy="80" r="74" fill="url(#mp-sheen)"/>
    </svg>
  </div>
  <div class="mp-hero-divider"></div>
  <div class="mp-hero-text">
    <div class="mp-hero-title">Mastery Path</div>
    <div class="mp-hero-sub">BOOTCAMP</div>
    <div class="mp-hero-rule"></div>
    <div class="mp-hero-byline">VINOD BHANUJA</div>
  </div>
</div>


A structured, module-wise Edge AI & TinyML training program that takes you
from "what is edge inference?" to designing production-grade, master-level
on-device machine-learning systems — training tiny models in Python,
converting and quantizing them with TensorFlow Lite (LiteRT), and running
them with TFLite-Micro on microcontrollers.

**No hardware required to start.** Every Level 1 module runs on a normal
laptop: you train models with Keras, convert and quantize them with the
TFLite converter, verify inference in Python, and even compile the
microcontroller-style C++ inference code as a desktop program.
Microcontroller deployment (ESP32/Arduino) is always shown as the follow-on
step — and the [Wokwi online simulator](https://wokwi.com/) covers much of it
in the browser.

## How the program is organized

| Level | Focus | Modules |
|-------|-------|---------|
| [Level 1 · Entry](level-1/index.md) | The edge AI workflow end-to-end: train → convert → quantize → deploy, TFLite-Micro, sensors & features, a gesture-recognition capstone | 9 topics + 1 capstone |
| [Level 2 · Intermediate](level-2/index.md) | Wake words, image classification on MCUs, Edge Impulse, pruning & distillation, ESP32-CAM projects | 9 topics + 1 project |
| [Level 3 · Advanced](level-3/index.md) | NPUs & accelerators (Coral, Ethos-U), ONNX Runtime, streaming inference, compilers, model security | 9 topics + 1 project |
| [Level 4 · Master](level-4/index.md) | Production edge fleets, OTA model updates, on-device learning, custom kernels & CMSIS-NN | 9 topics + 1 capstone |

## How to use this site

- Work through each level in order — later modules assume earlier ones.
- Every Level 1 topic page has real, runnable code — Python you can run in
  any environment with TensorFlow installed, and C/C++ you can compile on
  your own machine before ever touching a board.
- Each level ends with a project that combines everything learned in that
  level.
- Use the search bar (top of the page) to jump straight to a topic.

New to machine learning itself? The
[AI & ML Mastery Path](https://sigilipelli.github.io/ai-ml-mastery-path/)
covers model training fundamentals. New to microcontrollers? The
[Embedded Systems Mastery Path](https://sigilipelli.github.io/embedded-mastery-path/)
covers Arduino/ESP32 basics. Neither is strictly required to start here.

Start here → [Level 1 · Entry](level-1/index.md)

🎥 **Prefer video?** Watch the [Mastery Path video series](https://youtube.com/@sigilipelli) on YouTube — Shorts and full walkthroughs of these lessons.

## More from the Mastery Path series

Free, structured, module-wise training across 63 other languages, platforms and disciplines:

<div class="mastery-grid-wrap">
<p class="mastery-grid-category">Languages</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/python-mastery-path/">🐍 Python</a>
  <a href="https://sigilipelli.github.io/java-mastery-path/">☕ Java</a>
  <a href="https://sigilipelli.github.io/javascript-mastery-path/">🟨 JavaScript</a>
  <a href="https://sigilipelli.github.io/typescript-mastery-path/">🔷 TypeScript</a>
  <a href="https://sigilipelli.github.io/csharp-mastery-path/">🔵 C#</a>
  <a href="https://sigilipelli.github.io/shell-mastery-path/">🐚 Shell/Bash</a>
  <a href="https://sigilipelli.github.io/powershell-mastery-path/">💻 PowerShell</a>
  <a href="https://sigilipelli.github.io/c-mastery-path/">🇨 C</a>
  <a href="https://sigilipelli.github.io/cpp-mastery-path/">➕ C++</a>
  <a href="https://sigilipelli.github.io/go-mastery-path/">🐹 Go</a>
  <a href="https://sigilipelli.github.io/rust-mastery-path/">🦀 Rust</a>
  <a href="https://sigilipelli.github.io/sql-mastery-path/">🗄️ SQL</a>
  <a href="https://sigilipelli.github.io/ruby-mastery-path/">💎 Ruby</a>
  <a href="https://sigilipelli.github.io/php-mastery-path/">🐘 PHP</a>
  <a href="https://sigilipelli.github.io/kotlin-mastery-path/">🟣 Kotlin</a>
  <a href="https://sigilipelli.github.io/swift-mastery-path/">🐦 Swift</a>
  <a href="https://sigilipelli.github.io/dart-mastery-path/">🎯 Dart</a>
  <a href="https://sigilipelli.github.io/scala-mastery-path/">🔴 Scala</a>
  <a href="https://sigilipelli.github.io/r-mastery-path/">📊 R</a>
  <a href="https://sigilipelli.github.io/matlab-mastery-path/">🟧 MATLAB</a>
</div>
<p class="mastery-grid-category">Testing & QA</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/java-testing-mastery-path/">🧪 Java Testing</a>
  <a href="https://sigilipelli.github.io/cpp-testing-mastery-path/">🧪 C/C++ Testing</a>
  <a href="https://sigilipelli.github.io/python-testing-mastery-path/">🧪 Python Testing</a>
  <a href="https://sigilipelli.github.io/automotive-testing-mastery-path/">🚗 Automotive Testing</a>
</div>
<p class="mastery-grid-category">Security</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/cybersecurity-mastery-path/">🛡️ Cybersecurity</a>
</div>
<p class="mastery-grid-category">Cloud Platforms</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/aws-mastery-path/">☁️ AWS</a>
  <a href="https://sigilipelli.github.io/azure-mastery-path/">☁️ Azure</a>
  <a href="https://sigilipelli.github.io/gcp-mastery-path/">☁️ GCP</a>
  <a href="https://sigilipelli.github.io/ibm-cloud-mastery-path/">☁️ IBM Cloud</a>
  <a href="https://sigilipelli.github.io/adobe-mastery-path/">🎨 Adobe</a>
</div>
<p class="mastery-grid-category">Data & Analytics</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/data-engineering-mastery-path/">🛠️ Data Engineering</a>
  <a href="https://sigilipelli.github.io/data-science-mastery-path/">📈 Data Science</a>
  <a href="https://sigilipelli.github.io/tableau-mastery-path/">📊 Tableau</a>
  <a href="https://sigilipelli.github.io/excel-mastery-path/">📗 Excel</a>
  <a href="https://sigilipelli.github.io/pyspark-mastery-path/">⚡ PySpark</a>
  <a href="https://sigilipelli.github.io/etl-datalake-mastery-path/">🏞️ ETL & Data Lake</a>
</div>
<p class="mastery-grid-category">AI / ML / LLM</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/ai-ml-mastery-path/">🤖 AI/ML</a>
  <a href="https://sigilipelli.github.io/llm-dev-mastery-path/">🧠 LLM Dev</a>
  <a href="https://sigilipelli.github.io/rag-mastery-path/">📚 RAG</a>
  <a href="https://sigilipelli.github.io/claude-training-mastery-path/">🔶 Claude Training</a>
  <a href="https://sigilipelli.github.io/ai-tools-mastery-path/">🧰 AI Tools</a>
  <a href="https://sigilipelli.github.io/ml-math-mastery-path/">➗ ML Math Foundations</a>
</div>
<p class="mastery-grid-category">Embedded Systems</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/embedded-mastery-path/">🔌 Embedded</a>
  <a href="https://sigilipelli.github.io/embedded-linux-mastery-path/">🐧 Embedded Linux</a>
  <a href="https://sigilipelli.github.io/embedded-python-mastery-path/">🐍 Embedded Python</a>
  <a href="https://sigilipelli.github.io/freertos-mastery-path/">⏱️ FreeRTOS</a>
  <a href="https://sigilipelli.github.io/s32k-mastery-path/">🔧 S32K</a>
</div>
<p class="mastery-grid-category">Leadership & Management</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/product-manager-mastery-path/">📋 Product Manager</a>
  <a href="https://sigilipelli.github.io/product-lead-mastery-path/">🧭 Product Lead</a>
  <a href="https://sigilipelli.github.io/project-manager-mastery-path/">📅 Project Manager</a>
  <a href="https://sigilipelli.github.io/ai-manager-mastery-path/">🤖 AI Manager</a>
  <a href="https://sigilipelli.github.io/servant-leadership-mastery-path/">🤝 Servant Leadership</a>
</div>
<p class="mastery-grid-category">Professional Skills</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/english-fluency-mastery-path/">🗣️ English Fluency & IELTS</a>
  <a href="https://sigilipelli.github.io/workday-mastery-path/">🧑‍💼 Workday</a>
</div>
<p class="mastery-grid-category">Process & APIs</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/agile-mastery-path/">🔄 Agile/Scrum/Kanban</a>
  <a href="https://sigilipelli.github.io/rest-api-mastery-path/">🔗 REST API</a>
  <a href="https://sigilipelli.github.io/playwright-mastery-path/">🎭 Playwright</a>
</div>
<p class="mastery-grid-category">Infrastructure & Ops</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/server-ops-mastery-path/">🖥️ Server Ops</a>
  <a href="https://sigilipelli.github.io/nodemcu-mastery-path/">📶 NodeMCU/IoT</a>
  <a href="https://sigilipelli.github.io/terraform-mastery-path/">🌍 Terraform</a>
  <a href="https://sigilipelli.github.io/kubernetes-mastery-path/">☸️ Kubernetes</a>
  <a href="https://sigilipelli.github.io/docker-mastery-path/">🐳 Docker</a>
  <a href="https://sigilipelli.github.io/github-mastery-path/">🐙 GitHub & Git</a>
</div>
</div>
