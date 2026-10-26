const CLASS_COLORS = ["#e4572e", "#2f6fde", "#0e8f84"];
const CLASS_NAMES = ["Pothole", "Crack", "Manhole"];
const EXPERIMENTS = ["A_raw", "B_gaussian", "C_median", "D_clahe", "E_brightness", "F_combined"];
const ROUTES = {
  "/": { title: "Dự đoán ảnh", eyebrow: "PHÂN TÍCH ẢNH", heading: "Nhìn rõ từng dấu hiệu", subtitle: "Chạy phát hiện trên ảnh đường và đối chiếu với nhãn ground truth khi có." },
  "/video": { title: "Dự đoán video", eyebrow: "PHÂN TÍCH VIDEO", heading: "Theo dõi hư hỏng theo thời gian", subtitle: "Tải video lên, chọn model và xem kết quả phát hiện cùng các track đối tượng." },
  "/experiments": { title: "So sánh thí nghiệm", eyebrow: "PHÂN TÍCH THỰC NGHIỆM", heading: "Mỗi cấu hình, một góc nhìn", subtitle: "Đối chiếu các pipeline tiền xử lý bằng số liệu và trạng thái kiểm tra trực quan." },
  "/evaluation": { title: "Đánh giá mô hình", eyebrow: "ĐÁNH GIÁ", heading: "Đọc sâu hiệu năng mô hình", subtitle: "Xem chỉ số test, ma trận nhầm lẫn và phân tích theo từng lớp." },
};
const state = {
  route: "/",
  suite: "video",
  experiment: { image: "A_raw", video: "A_raw", evaluation: "A_raw" },
  matrixNormalized: false,
  videoSourceMode: "video",
  jobs: { image: null, video: null, evaluation: null },
  polling: new Set(),
  renderVersion: 0,
  comparisonSuite: "video",
  evaluationSuite: "video",
};

const app = document.getElementById("app");
const breadcrumb = document.getElementById("breadcrumb-current");

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "") : "—";
}

function formatPercent(value) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
}

function routeFromHash() {
  const path = window.location.hash.replace(/^#/, "") || "/";
  return Object.hasOwn(ROUTES, path) ? path : "/";
}

function updateNavigation() {
  const page = ROUTES[state.route];
  breadcrumb.textContent = page.title;
  document.title = `${page.title} — RoadScan`;
  document.querySelectorAll(".nav-link").forEach((link) => {
    if (link.dataset.route === state.route) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Yêu cầu thất bại (${response.status})`);
  return data;
}

function heading(route, extra = "") {
  const page = ROUTES[route];
  return `<div class="page-heading">
    <div><p class="eyebrow">${page.eyebrow}</p><h1>${page.heading}</h1><p class="page-subtitle">${page.subtitle}</p></div>
    ${extra}
  </div>`;
}

function card(title, description, content, className = "") {
  return `<section class="card card-pad ${className}">
    <div class="card-head"><div><h2 class="card-title">${title}</h2>${description ? `<p class="card-desc">${description}</p>` : ""}</div></div>
    ${content}
  </section>`;
}

function emptyState(title, description, icon = "◫") {
  return `<div class="empty-state"><span class="empty-art" aria-hidden="true">${icon}</span><strong>${title}</strong><p>${description}</p></div>`;
}

function experimentOptions(experiments, selected) {
  if (!experiments.length) return `<option value="">Chưa có model khả dụng</option>`;
  return experiments.map((item) =>
    `<option value="${escapeHTML(item.name)}" ${item.name === selected ? "selected" : ""}>${escapeHTML(item.name)}</option>`
  ).join("");
}

function experimentSelect(experiments, selected, id, label = "Mô hình thí nghiệm") {
  return `<label class="field"><span class="field-label">${label}</span>
    <select class="input experiment-select" id="${id}" ${experiments.length ? "" : "disabled"}>${experimentOptions(experiments, selected)}</select>
    <span class="field-help">${experiments.length ? `${experiments.length} model có checkpoint hoàn chỉnh` : "Chưa tìm thấy best.pt và experiment_config.yaml trong thư mục thí nghiệm."}</span>
  </label>`;
}

function filePicker({ id, label, multiple = false, accept, title, description }) {
  return `<div class="field">
    <span class="field-label">${label}</span>
    <label class="dropzone" for="${id}" data-dropzone="${id}">
      <span class="drop-icon" aria-hidden="true">↑</span>
      <span class="drop-title">${title}</span><span class="drop-meta">${description}</span>
    </label>
    <input class="file-input" id="${id}" type="file" accept="${accept}" ${multiple ? "multiple" : ""}>
    <div class="file-list" id="${id}-list" aria-live="polite"></div>
  </div>`;
}

function toggle(label, description, id, pressed) {
  return `<div class="toggle-row"><span class="toggle-copy"><strong>${label}</strong><small>${description}</small></span>
    <button class="switch" type="button" id="${id}" aria-label="${label}" aria-pressed="${pressed}"></button></div>`;
}

function commandPreview(id, command) {
  return `<div class="command-box"><div class="command-head"><span>Lệnh dự kiến</span><button class="text-button" type="button" data-copy="${id}">Sao chép</button></div><code id="${id}">${escapeHTML(command)}</code></div>`;
}

function selectedExperiment(type, experiments) {
  const chosen = state.experiment[type];
  if (experiments.some((item) => item.name === chosen)) return chosen;
  return experiments[0]?.name || "";
}

function updateFileList(input) {
  const list = document.getElementById(`${input.id}-list`);
  if (!list) return;
  const files = [...input.files];
  list.innerHTML = files.slice(0, 5).map((file) =>
    `<div class="file-line"><span>${escapeHTML(file.name)}</span><span>${(file.size / 1024 / 1024).toFixed(1)} MB</span></div>`
  ).join("") + (files.length > 5 ? `<span class="field-help">và ${files.length - 5} tệp khác</span>` : "");
}

function connectDropzone(id) {
  const input = document.getElementById(id);
  const zone = document.querySelector(`[data-dropzone="${id}"]`);
  if (!input || !zone) return;
  input.addEventListener("change", () => updateFileList(input));
  for (const eventName of ["dragenter", "dragover"]) {
    zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.add("dragover"); });
  }
  for (const eventName of ["dragleave", "drop"]) {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
      if (eventName === "drop" && event.dataTransfer.files.length) {
        input.files = event.dataTransfer.files;
        updateFileList(input);
      }
    });
  }
}

function connectToggle(id, onChange) {
  const button = document.getElementById(id);
  if (!button) return;
  button.addEventListener("click", () => {
    const value = button.getAttribute("aria-pressed") !== "true";
    button.setAttribute("aria-pressed", String(value));
    onChange?.(value);
  });
}

function connectCopyButtons() {
  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const text = document.getElementById(button.dataset.copy)?.textContent || "";
      try {
        await navigator.clipboard.writeText(text);
        const oldText = button.textContent;
        button.textContent = "Đã sao chép";
        setTimeout(() => { button.textContent = oldText; }, 1400);
      } catch {
        button.textContent = "Không thể sao chép";
      }
    });
  });
}

function updatePreviewCommand(kind) {
  const target = document.getElementById(`${kind}-command`);
  if (!target) return;
  const input = document.getElementById(`${kind}-files`);
  const experiment = document.getElementById(`${kind}-experiment`)?.value || "A_raw";
  const file = input?.files?.[0]?.name || "<chọn-tệp>";
  const confidence = document.getElementById("image-conf")?.value || "0.25";
  let command;
  if (kind === "image") {
    const comparison = document.getElementById("image-gt")?.getAttribute("aria-pressed") === "true" ? " --show-ground-truth" : "";
    const side = document.getElementById("image-side")?.getAttribute("aria-pressed") === "true" ? " --side-by-side" : "";
    command = `python predict.py --experiment experiments/yolo_video/${experiment} --source "${file}" --conf ${confidence} --max-images ${document.getElementById("image-max")?.value || 20}${comparison}${side}`;
  } else {
    const source = state.videoSourceMode === "webcam" ? document.getElementById("camera-index")?.value || "0" : `"${file}"`;
    const frameLimit = state.videoSourceMode === "webcam" ? ` --max-frames ${document.getElementById("camera-frames")?.value || 300}` : "";
    command = `python predict.py --experiment experiments/yolo_video/${experiment} --video ${source} --frame-step ${document.getElementById("video-step")?.value || 1} --frame-size ${document.getElementById("video-size")?.value || 640} --min-track-frames ${document.getElementById("video-track")?.value || 3}${frameLimit}`;
  }
  target.textContent = command;
}

function statusMarkup(job, targetType) {
  if (!job) return "";
  const busy = ["queued", "running"].includes(job.status);
  const statusText = job.status === "queued" ? "Đang chờ xử lý…" :
    job.status === "running" ? "Đang chạy pipeline…" :
      job.status === "completed" ? "Hoàn tất" : "Có lỗi khi chạy";
  return `<div class="job-status" id="${targetType}-job-status">${busy ? '<span class="spinner" aria-hidden="true"></span>' : job.status === "completed" ? '<span class="status-pip"></span>' : "!"}
    <span>${statusText}${job.error ? ` — ${escapeHTML(job.error)}` : ""}</span></div>
    ${job.log ? `<pre class="log-box">${escapeHTML(job.log)}</pre>` : ""}`;
}

function resultMarkup(job, kind) {
  if (!job || job.status !== "completed") return "";
  const summary = job.summary || {};
  if (kind === "video") {
    const video = job.files.find((file) => file.name.toLowerCase().endsWith(".mp4"));
    const details = summary.videos?.[0] || {};
    const counts = details.unique_objects_by_class || {};
    return `<div class="stack">
      ${video ? `<video class="video-preview" controls preload="metadata" src="${escapeHTML(video.url)}"></video>
        <a class="subtle-link" href="${escapeHTML(video.url)}" download>Tải video đã gắn nhãn</a>` : ""}
      <div class="video-stat-grid">
        ${["pothole", "crack", "manhole"].map((name, index) => `<div class="video-stat"><span><i class="class-dot ${index ? index === 1 ? "crack" : "manhole" : ""}"></i> ${name}</span><strong>${summary.unique_objects_by_class?.[name] ?? "—"}</strong></div>`).join("")}
      </div>
      <div class="result-meta"><span class="meta-pill">${details.frames_read ?? "—"} frame</span><span class="meta-pill">${details.frames_processed ?? "—"} đã xử lý</span><span class="meta-pill">${details.duration_s ?? "—"} giây</span><span class="meta-pill">${details.processing_fps ?? "—"} FPS</span></div>
      ${renderTimeline(details.confirmed_tracks || [], details.duration_s)}
      ${job.files.find((file) => file.name === "video_summary.json") ? `<a class="subtle-link" href="${escapeHTML(job.files.find((file) => file.name === "video_summary.json").url)}" download>Tải video_summary.json</a>` : ""}
    </div>`;
  }
  const images = job.files.filter((file) => /\.(jpg|jpeg|png)$/i.test(file.name));
  const summaryFile = job.files.find((file) => file.name === "prediction_summary.json");
  const firstImage = images[0];
  const compare = summary.images?.flatMap((image) => Object.values(image.comparison || {})) || [];
  const totals = compare.reduce((result, item) => {
    for (const key of ["tp", "fp", "fn"]) result[key] += Number(item[key] || 0);
    return result;
  }, { tp: 0, fp: 0, fn: 0 });
  return `<div class="stack">
    ${firstImage ? `<img class="result-preview" src="${escapeHTML(firstImage.url)}" alt="Ảnh sau dự đoán">` : ""}
    <div class="thumbnail-strip">${images.map((image) => `<a class="thumbnail-link" href="${escapeHTML(image.url)}" target="_blank" rel="noreferrer"><img src="${escapeHTML(image.url)}" alt=""><span>${escapeHTML(image.name)}</span></a>`).join("")}</div>
    ${summary.mode === "prediction_with_ground_truth" ? `<div class="result-meta"><span class="meta-pill">GT: ${totals.tp + totals.fn}</span><span class="meta-pill">Pred: ${totals.tp + totals.fp}</span><span class="meta-pill">TP ${totals.tp}</span><span class="meta-pill">FP ${totals.fp}</span><span class="meta-pill">FN ${totals.fn}</span></div>` : ""}
    <div class="result-meta"><span class="meta-pill">${summary.processed_images ?? "—"} ảnh đã xử lý</span><span class="meta-pill">${escapeHTML(summary.experiment || "—")}</span><span class="meta-pill">conf ${summary.confidence_threshold ?? "—"}</span></div>
    ${summaryFile ? `<a class="subtle-link" href="${escapeHTML(summaryFile.url)}" download>Tải prediction_summary.json</a>` : ""}
  </div>`;
}

function renderTimeline(tracks, duration) {
  if (!tracks.length) return `<p class="notice">Chưa có track đủ số frame để hiển thị dòng thời gian.</p>`;
  const end = Math.max(Number(duration) || 0, 0.001);
  return `<section><div class="section-kicker">VÒNG ĐỜI TRACK ĐÃ XÁC NHẬN</div><div class="timeline" style="margin-top:12px">
    ${tracks.map((track) => {
      const left = Math.max(0, Math.min(100, Number(track.first_seen_s || 0) / end * 100));
      const width = Math.max(1, Math.min(100 - left, (Number(track.last_seen_s || 0) - Number(track.first_seen_s || 0)) / end * 100));
      return `<div class="timeline-row"><span>#${track.track_id} ${escapeHTML(track.class_name)}</span><div class="timeline-track"><span class="timeline-segment" style="left:${left}%;width:${width}%;background:${CLASS_COLORS[track.class_id] || CLASS_COLORS[0]}"></span></div></div>`;
    }).join("")}
  </div></section>`;
}

async function renderImagePage(renderVersion) {
  const experiments = (await api("/api/experiments?suite=video")).experiments;
  if (renderVersion !== state.renderVersion) return;
  const selected = selectedExperiment("image", experiments);
  state.experiment.image = selected;
  const job = state.jobs.image;
  app.innerHTML = `${heading("/")}
    <div class="layout-two">
      <div class="stack">
        <section class="card card-pad">
          <div class="card-head"><div><h2 class="card-title">Cấu hình dự đoán</h2><p class="card-desc">Chọn model và ảnh cần phân tích.</p></div><span class="section-kicker">01 — INPUT</span></div>
          ${experimentSelect(experiments, selected, "image-experiment")}
          ${filePicker({ id: "image-files", label: "Ảnh đầu vào", multiple: true, accept: ".jpg,.jpeg,.png,image/jpeg,image/png", title: "Chọn hoặc thả ảnh vào đây", description: "JPG, JPEG, PNG · có thể chọn nhiều ảnh" })}
          <div class="field"><label class="field-label" for="image-conf">Ngưỡng confidence</label><div class="range-row"><input id="image-conf" type="range" min="0.05" max="0.95" step="0.05" value="0.25"><output id="image-conf-label" class="range-number">0.25</output></div></div>
          <div class="field"><label class="field-label" for="image-max">Giới hạn số ảnh</label><input class="input" id="image-max" type="number" min="1" max="500" value="20"></div>
          ${toggle("So sánh ground truth", "Hiển thị GT nếu ảnh có nhãn tương ứng.", "image-gt", true)}
          ${toggle("Hiển thị song song", "Tách ảnh GT và dự đoán thành hai panel.", "image-side", false)}
          <button class="button primary button-full" id="run-image" ${experiments.length ? "" : "disabled"}>▶ <span>Chạy dự đoán ảnh</span></button>
          <p class="field-help">Lưu ý: chạy mặc định sẽ tạo thư mục kết quả mới, không xoá kết quả cũ.</p>
          ${commandPreview("image-command", `python predict.py --experiment experiments/yolo_video/${selected || "A_raw"} --source "<chọn-ảnh>" --max-images 20 --show-ground-truth`)}
          <div id="image-job">${statusMarkup(job, "image")}</div>
        </section>
      </div>
      <div class="stack">
        <section class="card card-pad"><div class="card-head"><div><h2 class="card-title">Kết quả dự đoán</h2><p class="card-desc">Ảnh xem trước, nhãn và báo cáo của lần chạy gần nhất.</p></div><span class="section-kicker">02 — OUTPUT</span></div>
          <div id="image-result">${job?.status === "completed" ? resultMarkup(job, "image") : emptyState("Sẵn sàng phân tích", "Chọn một hoặc nhiều ảnh rồi chạy dự đoán. Ảnh kết quả và prediction_summary.json sẽ xuất hiện tại đây.", "⌕")}</div>
        </section>
        ${card("Chú giải", "Tên lớp và tiền tố luôn hiển thị cùng màu để dễ phân biệt.", `<div class="result-meta"><span class="meta-pill"><span class="class-dot pothole"></span> 0 · Pothole</span><span class="meta-pill"><span class="class-dot crack"></span> 1 · Crack</span><span class="meta-pill"><span class="class-dot manhole"></span> 2 · Manhole</span><span class="meta-pill">GT: ground truth</span><span class="meta-pill">Pred: dự đoán</span></div><p class="footer-note">Chỉ số TP / FP / FN là phép so khớp nhanh IoU ≥ 0.5; không thay thế mAP đánh giá trên tập test.</p>`)}
      </div>
    </div>`;
  connectDropzone("image-files");
  document.getElementById("image-experiment").addEventListener("change", (event) => { state.experiment.image = event.target.value; updatePreviewCommand("image"); });
  document.getElementById("image-files").addEventListener("change", () => updatePreviewCommand("image"));
  document.getElementById("image-conf").addEventListener("input", (event) => {
    document.getElementById("image-conf-label").value = Number(event.target.value).toFixed(2);
    updatePreviewCommand("image");
  });
  document.getElementById("image-conf").addEventListener("change", () => updatePreviewCommand("image"));
  document.getElementById("image-max").addEventListener("input", () => updatePreviewCommand("image"));
  connectToggle("image-gt", updatePreviewCommand.bind(null, "image"));
  connectToggle("image-side", updatePreviewCommand.bind(null, "image"));
  connectCopyButtons();
  document.getElementById("run-image").addEventListener("click", () => runPrediction("image"));
}

async function renderVideoPage(renderVersion) {
  const experiments = (await api("/api/experiments?suite=video")).experiments;
  if (renderVersion !== state.renderVersion) return;
  const selected = selectedExperiment("video", experiments);
  state.experiment.video = selected;
  const job = state.jobs.video;
  app.innerHTML = `${heading("/video")}
    <div class="banner" role="note" style="margin-bottom:16px"><span class="banner-icon">!</span><div><strong>Phạm vi model video</strong><p>Bộ dữ liệu video hiện chỉ có nhãn <b>pothole</b>. Model không được huấn luyện để nhận diện crack hoặc manhole; số đếm cho hai lớp đó không đại diện cho khả năng phát hiện.</p></div></div>
    <div class="layout-two">
      <div class="stack"><section class="card card-pad">
        <div class="card-head"><div><h2 class="card-title">Cấu hình video</h2><p class="card-desc">Video được tải lên máy cục bộ và xử lý bằng model đã chọn.</p></div><span class="section-kicker">01 — INPUT</span></div>
        ${experimentSelect(experiments, selected, "video-experiment")}
        <div class="segmented source-switch" role="group" aria-label="Nguồn video">
          <button type="button" data-video-source="video" aria-pressed="${state.videoSourceMode === "video"}">Video tải lên</button>
          <button type="button" data-video-source="webcam" aria-pressed="${state.videoSourceMode === "webcam"}">Webcam</button>
        </div>
        <div id="video-file-panel" class="${state.videoSourceMode === "webcam" ? "hidden" : ""}">
          ${filePicker({ id: "video-files", label: "Video đầu vào", accept: ".mp4,.avi,.mov,.mkv,.wmv,.webm,.m4v,video/*", title: "Chọn hoặc thả một video", description: "MP4, AVI, MOV, MKV, WEBM · tối đa 1 GiB" })}
        </div>
        <div id="webcam-panel" class="form-grid ${state.videoSourceMode === "webcam" ? "" : "hidden"}">
          <div class="field"><label class="field-label" for="camera-index">Camera index</label><input class="input" id="camera-index" type="number" min="0" max="16" value="0"></div>
          <div class="field"><label class="field-label" for="camera-frames">Giới hạn frame</label><input class="input" id="camera-frames" type="number" min="1" max="1000000" value="300"><span class="field-help">Cần giới hạn để kết thúc phiên webcam.</span></div>
        </div>
        <div class="form-grid">
          <div class="field"><label class="field-label" for="video-step">Frame step</label><input class="input" id="video-step" type="number" min="1" max="1000" value="1"><span class="field-help">1 = xử lý mọi frame.</span></div>
          <div class="field"><label class="field-label" for="video-size">Frame size</label><input class="input" id="video-size" type="number" min="0" max="4096" value="640"><span class="field-help">0 = giữ kích thước gốc.</span></div>
          <div class="field span-two"><label class="field-label" for="video-track">Số frame xác nhận track</label><input class="input" id="video-track" type="number" min="1" max="1000" value="3"></div>
        </div>
        ${toggle("Lưu chi tiết từng frame", "Ghi detections vào video_summary.json.", "video-details", true)}
        <button class="button primary button-full" id="run-video" ${experiments.length ? "" : "disabled"}>▷ <span>Chạy phát hiện video</span></button>
        ${commandPreview("video-command", `python predict.py --experiment experiments/yolo_video/${selected || "A_raw"} --video "<chọn-video>" --frame-step 1 --frame-size 640 --min-track-frames 3`)}
        <div id="video-job">${statusMarkup(job, "video")}</div>
      </section></div>
      <div class="stack"><section class="card card-pad">
        <div class="card-head"><div><h2 class="card-title">Kết quả video</h2><p class="card-desc">Video đã gắn nhãn, bộ đếm track và tóm tắt xử lý.</p></div><span class="section-kicker">02 — OUTPUT</span></div>
        <div id="video-result">${job?.status === "completed" ? resultMarkup(job, "video") : emptyState("Chưa có video trong phiên này", "Sau khi chạy, video kết quả có thể xem và tải trực tiếp tại đây.", "▷")}</div>
      </section>
      ${card("Cách đọc track", "Bộ đếm là số đối tượng duy nhất được theo dõi xuyên suốt video.", `<div class="result-meta"><span class="meta-pill">#ID · mã track</span><span class="meta-pill">lớp đã bỏ phiếu</span><span class="meta-pill">frame xác nhận</span></div><p class="footer-note">Video người dùng tải lên không có ground truth đi kèm, nên giao diện không tính precision/recall cho video đó.</p>`)}</div>
    </div>`;
  connectDropzone("video-files");
  document.querySelectorAll("[data-video-source]").forEach((button) => button.addEventListener("click", () => {
    state.videoSourceMode = button.dataset.videoSource;
    document.querySelectorAll("[data-video-source]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    document.getElementById("video-file-panel").classList.toggle("hidden", state.videoSourceMode !== "video");
    document.getElementById("webcam-panel").classList.toggle("hidden", state.videoSourceMode !== "webcam");
    updatePreviewCommand("video");
  }));
  document.getElementById("video-experiment").addEventListener("change", (event) => { state.experiment.video = event.target.value; updatePreviewCommand("video"); });
  document.getElementById("video-files").addEventListener("change", () => updatePreviewCommand("video"));
  for (const id of ["video-step", "video-size", "video-track", "camera-index", "camera-frames"]) {
    document.getElementById(id)?.addEventListener("input", () => updatePreviewCommand("video"));
  }
  connectToggle("video-details");
  connectCopyButtons();
  document.getElementById("run-video").addEventListener("click", () => runPrediction("video"));
}

async function runPrediction(kind) {
  const input = document.getElementById(`${kind}-files`);
  const files = [...(input?.files || [])];
  const webcam = kind === "video" && state.videoSourceMode === "webcam";
  if (!files.length && !webcam) {
    window.alert(kind === "image" ? "Hãy chọn ít nhất một ảnh." : "Hãy chọn một video.");
    return;
  }
  if (kind === "video" && !webcam && files.length !== 1) {
    window.alert("Mỗi lần chỉ xử lý một video.");
    return;
  }
  const form = new FormData();
  form.append("suite", "video");
  form.append("experiment", document.getElementById(`${kind}-experiment`).value);
  form.append("conf", document.getElementById("image-conf")?.value || "0.25");
  for (const file of files) form.append("files", file, file.name);
  if (kind === "image") {
    form.append("max_images", document.getElementById("image-max").value);
    form.append("show_ground_truth", String(document.getElementById("image-gt").getAttribute("aria-pressed") === "true"));
    form.append("side_by_side", String(document.getElementById("image-side").getAttribute("aria-pressed") === "true"));
  } else {
    form.append("source_mode", state.videoSourceMode);
    form.append("frame_step", document.getElementById("video-step").value);
    form.append("frame_size", document.getElementById("video-size").value);
    form.append("min_track_frames", document.getElementById("video-track").value);
    if (webcam) {
      form.append("camera_index", document.getElementById("camera-index").value);
      form.append("max_frames", document.getElementById("camera-frames").value);
    }
    form.append("save_frame_detections", String(document.getElementById("video-details").getAttribute("aria-pressed") === "true"));
  }
  const status = document.getElementById(`${kind}-job`);
  const result = document.getElementById(`${kind}-result`);
  const button = document.getElementById(`run-${kind}`);
  button.disabled = true;
  state.jobs[kind] = { status: "queued", log: "" };
  status.innerHTML = statusMarkup(state.jobs[kind], kind);
  try {
    const response = await api(`/api/predict/${kind}`, { method: "POST", body: form });
    state.jobs[kind].id = response.job_id;
    await watchJob(kind, response.job_id);
  } catch (error) {
    state.jobs[kind] = { status: "failed", error: error.message };
    status.innerHTML = statusMarkup(state.jobs[kind], kind);
  } finally {
    button.disabled = false;
    if (result && state.jobs[kind]?.status === "completed") result.innerHTML = resultMarkup(state.jobs[kind], kind);
  }
}

async function watchJob(kind, id) {
  if (state.polling.has(id)) return;
  state.polling.add(id);
  try {
    while (true) {
      const job = await api(`/api/jobs/${id}`);
      state.jobs[kind] = job;
      if (state.route === (kind === "video" ? "/video" : "/")) {
        const status = document.getElementById(`${kind}-job`);
        if (status) status.innerHTML = statusMarkup(job, kind);
        if (job.status === "completed" || job.status === "failed") {
          const result = document.getElementById(`${kind}-result`);
          if (result && job.status === "completed") result.innerHTML = resultMarkup(job, kind);
          if (result && job.status === "failed") result.innerHTML = `<div class="banner danger"><span class="banner-icon">!</span><div><strong>Không chạy được pipeline</strong><p>${escapeHTML(job.error)}</p></div></div>`;
        }
      }
      if (!["queued", "running"].includes(job.status)) break;
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
  } catch (error) {
    state.jobs[kind] = { ...state.jobs[kind], status: "failed", error: error.message };
  } finally {
    state.polling.delete(id);
  }
}

function suiteSwitcher(suite, key) {
  return `<div class="segmented" role="group" aria-label="Chọn bộ thí nghiệm">
    <button type="button" data-suite="${key}" data-value="image" aria-pressed="${suite === "image"}">Bộ ảnh</button>
    <button type="button" data-suite="${key}" data-value="video" aria-pressed="${suite === "video"}">Bộ video</button>
  </div>`;
}

function connectSuiteSwitcher(key, setter) {
  document.querySelectorAll(`[data-suite="${key}"]`).forEach((button) => button.addEventListener("click", () => {
    setter(button.dataset.value);
    render();
  }));
}

function renderExperimentTable(rows) {
  if (!rows.length) return emptyState("Chưa có bảng kết quả", "Chưa tìm thấy experiment_results.csv cho bộ dữ liệu này.");
  return `<div class="table-wrap"><table class="data-table"><thead><tr>
    <th>THÍ NGHIỆM</th><th>TIỀN XỬ LÝ</th><th>FRACTION</th><th>IMGSZ</th><th>PRECISION</th><th>RECALL</th><th>MAP50</th><th>MAP50–95</th><th>PREP MS</th><th>INFER MS</th><th>TOTAL MS</th>
  </tr></thead><tbody>${rows.map((row) => `<tr>
    <td><span class="experiment-name"><i class="exp-marker"></i>${escapeHTML(row.experiment)}</span></td><td>${escapeHTML(row.preprocessing || "—")}</td>
    <td class="table-number">${formatNumber(row.fraction, 1)}</td><td class="table-number">${escapeHTML(row.imgsz || "—")}</td>
    <td class="table-number">${formatNumber(row.precision)}</td><td class="table-number">${formatNumber(row.recall)}</td>
    <td class="table-number">${formatNumber(row.mAP50)}</td><td class="table-number">${formatNumber(row.mAP50_95)}</td>
    <td class="table-number">${formatNumber(row.preprocessing_ms, 2)}</td><td class="table-number">${formatNumber(row.inference_ms, 2)}</td><td class="table-number">${formatNumber(row.total_ms, 2)}</td>
  </tr>`).join("")}</tbody></table></div>`;
}

function renderVisualProgress(rows) {
  if (!rows.length) return `<p class="notice">Chưa có visual_inspection.csv; trạng thái kiểm tra trực quan chưa khả dụng.</p>`;
  return `<div class="experiment-rows">${rows.map((row) => {
    const count = Math.max(0, Number(row.reviewed_images) || 0);
    const percent = Math.min(100, count / 10 * 100);
    const complete = row.status === "COMPLETE";
    return `<div><div class="progress-head"><span>${escapeHTML(row.experiment)}</span><span>${count} / 10 · ${complete ? "COMPLETE" : "PENDING"}</span></div><div class="progress-track"><div class="progress-fill" style="width:${percent}%"></div></div></div>`;
  }).join("")}</div>`;
}

function renderMetricChart(rows) {
  if (!rows.length) return emptyState("Chưa có số liệu", "Biểu đồ sẽ hiển thị sau khi pipeline sinh bảng experiment_results.csv.", "⌁");
  return `<div class="chart-bars">${rows.map((row) => {
    const value = Number(row.mAP50);
    return `<div class="chart-row"><span>${escapeHTML(row.experiment)}</span><div class="chart-track"><div class="chart-fill" style="width:${Number.isFinite(value) ? Math.max(0, Math.min(100, value * 100)) : 0}%"></div></div><span class="table-number">${formatNumber(row.mAP50)}</span></div>`;
  }).join("")}</div><p class="footer-note">Thanh biểu diễn mAP50. Chưa gắn nhãn phương pháp nào là tốt nhất.</p>`;
}

async function renderExperimentsPage(renderVersion) {
  const suite = state.comparisonSuite;
  const data = await api(`/api/results/experiments?suite=${suite}`);
  if (renderVersion !== state.renderVersion) return;
  const reasons = data.analysis?.blocking_reasons || [];
  const ready = data.analysis?.conclusion_status === "READY";
  app.innerHTML = `${heading("/experiments", suiteSwitcher(suite, "comparison"))}
    <section class="readiness ${ready ? "ready" : ""}" style="margin-bottom:16px"><h3>${ready ? "ĐÃ ĐỦ ĐIỀU KIỆN PHÂN TÍCH" : "BLOCKED — CHƯA ĐỦ BẰNG CHỨNG"}</h3>
      <ul>${(reasons.length ? reasons : [
        "Sáu thí nghiệm cần cùng cấu hình huấn luyện và fraction = 1.0.",
        "Cần đủ chỉ số tổng thể, theo lớp và độ trễ cho mọi run.",
        "Cần hoàn tất kiểm tra trực quan ít nhất 10 ảnh mỗi thí nghiệm.",
      ]).map((reason) => `<li>${escapeHTML(reason)}</li>`).join("")}</ul>
    </section>
    <div class="split-layout">
      ${card("Tiến độ kiểm tra trực quan", "Trạng thái đọc từ visual_inspection.csv; không tự đánh dấu hoàn tất.", renderVisualProgress(data.visual_inspection))}
      ${card("mAP50 trên tập test", "Tham khảo xu hướng; không tuyên bố phương pháp chiến thắng.", renderMetricChart(data.rows))}
    </div>
    <section class="card card-pad" style="margin-top:16px"><div class="card-head"><div><h2 class="card-title">Bảng đối chiếu</h2><p class="card-desc">Số liệu thực tế từ experiment_results.csv.</p></div></div>
      ${renderExperimentTable(data.rows)}
    </section>
    <p class="footer-note">Nếu độ trễ được đo trên GPU Colab, không so trực tiếp với tốc độ trên CPU cục bộ.</p>`;
  connectSuiteSwitcher("comparison", (value) => { state.comparisonSuite = value; });
}

function classMetricRows(metrics, csvRows) {
  if (csvRows?.length) return csvRows;
  return (metrics?.per_class || []).map((item) => ({
    class: item.class_name,
    precision: item.precision,
    recall: item.recall,
    mAP50: item.mAP50,
    mAP50_95: item.mAP50_95,
  }));
}

function renderClassMetrics(metrics, csvRows) {
  const rows = classMetricRows(metrics, csvRows);
  if (!rows.length) return emptyState("Chưa có metric theo lớp", "Chạy đánh giá để tạo kết quả theo từng lớp.");
  return `<div class="table-wrap"><table class="data-table"><thead><tr><th>LỚP</th><th>PRECISION</th><th>RECALL</th><th>MAP50</th><th>MAP50–95</th></tr></thead><tbody>
  ${rows.map((row) => `<tr><td>${escapeHTML(row.class || row.class_name || row.name || "—")}</td><td class="table-number">${formatNumber(row.precision)}</td><td class="table-number">${formatNumber(row.recall)}</td><td class="table-number">${formatNumber(row.mAP50)}</td><td class="table-number">${formatNumber(row.mAP50_95)}</td></tr>`).join("")}
  </tbody></table></div>`;
}

function matrixMarkup(matrix, normalized) {
  if (!matrix?.counts?.length) return emptyState("Chưa có ma trận nhầm lẫn", "Không tìm thấy ma trận trong báo cáo evaluation.");
  const labels = matrix.labels || ["pothole", "crack", "manhole", "background"];
  const counts = matrix.counts;
  const columnTotals = labels.map((_, col) => counts.reduce((sum, row) => sum + Number(row[col] || 0), 0));
  return `<div class="table-wrap"><table class="matrix"><thead><tr><th>Dự đoán ↓<br>Thực tế →</th>${labels.map((label) => `<th>${escapeHTML(label)}</th>`).join("")}</tr></thead>
    <tbody>${counts.map((row, rowIndex) => `<tr><th>${escapeHTML(labels[rowIndex] || "—")}</th>${row.map((value, colIndex) => {
      const fraction = columnTotals[colIndex] ? Number(value) / columnTotals[colIndex] : 0;
      const shown = normalized ? `${(fraction * 100).toFixed(1)}%` : formatNumber(value, 0);
      const hot = (rowIndex === 0 && colIndex === 2) || (rowIndex === 2 && colIndex === 0) || (rowIndex === 1 && colIndex === 3) || (rowIndex === 0 && colIndex === 3);
      return `<td class="${hot && Number(value) ? "hot" : ""}" title="${formatPercent(fraction)} theo cột">${shown}</td>`;
    }).join("")}</tr>`).join("")}</tbody>
  </table></div><p class="footer-note">Hàng là lớp dự đoán, cột là ground truth. Chuẩn hoá theo tổng từng cột.</p>`;
}

function renderBalance(report) {
  const counts = report?.bbox_statistics_by_imgsz?.["320"]?.overall?.size_counts;
  const classCounts = report?.class_imbalance?.per_split_objects?.train;
  if (!counts || !classCounts) return emptyState("Chưa có báo cáo cân bằng lớp", "Chưa tìm thấy báo cáo class balance trong results/analysis/class_balance.");
  const total = Object.values(classCounts).reduce((sum, value) => sum + Number(value || 0), 0);
  const values = [
    ["Pothole", Number(classCounts.pothole || 0), CLASS_COLORS[0]],
    ["Crack", Number(classCounts.crack || 0), CLASS_COLORS[1]],
    ["Manhole", Number(classCounts.manhole || 0), CLASS_COLORS[2]],
  ];
  const smallRatio = Number(counts.small || 0) / Math.max(1, Object.values(counts).reduce((sum, value) => sum + Number(value || 0), 0));
  return `<div class="chart-bars">${values.map(([name, count, color]) => `<div class="chart-row"><span>${name}</span><div class="chart-track"><div class="chart-fill" style="width:${total ? count / total * 100 : 0}%;background:${color}"></div></div><span class="table-number">${count}</span></div>`).join("")}</div>
    <div class="class-bar" style="margin-top:14px">${values.map(([, count, color]) => `<span style="width:${total ? count / total * 100 : 0}%;background:${color}" title="${count} objects"></span>`).join("")}</div>
    <div class="result-meta"><span class="meta-pill">Small ${counts.small ?? "—"}</span><span class="meta-pill">Medium ${counts.medium ?? "—"}</span><span class="meta-pill">Large ${counts.large ?? "—"}</span><span class="meta-pill">imgsz 320</span></div>
    ${smallRatio >= .7 ? `<div class="banner" style="margin-top:14px"><span class="banner-icon">!</span><div><strong>Nhiều box nhỏ</strong><p>${formatPercent(smallRatio)} box thuộc nhóm small. Đây là thông tin mô tả; không tự bật oversampling hoặc tăng trọng số lớp.</p></div></div>` : ""}
    <p class="footer-note">Số lượng lớp lấy từ báo cáo ảnh hiện có; thống kê kích thước box dùng ngưỡng COCO tại 320px.</p>`;
}

function connectMatrixToggle() {
  document.querySelectorAll("[data-matrix-mode]").forEach((button) => button.addEventListener("click", () => {
    state.matrixNormalized = button.dataset.matrixMode === "normalized";
    render();
  }));
}

function evaluationJobMarkup(job) {
  if (!job) return "";
  const done = job.status === "completed";
  const failed = job.status === "failed";
  return `<div class="job-status">${done ? '<span class="status-pip"></span>' : failed ? "!" : '<span class="spinner"></span>'}<span>${done ? "Đánh giá hoàn tất. Tải lại trang đánh giá để xem số liệu mới." : failed ? escapeHTML(job.error || "Đánh giá thất bại") : "Đang chạy đánh giá tập test…"}</span></div>${job.log ? `<pre class="log-box">${escapeHTML(job.log)}</pre>` : ""}`;
}

async function renderEvaluationPage(renderVersion) {
  const suite = state.evaluationSuite;
  const experiments = (await api(`/api/experiments?suite=${suite}`)).experiments;
  if (renderVersion !== state.renderVersion) return;
  const selected = experiments.some((item) => item.name === state.experiment.evaluation) ? state.experiment.evaluation : (experiments[0]?.name || "");
  state.experiment.evaluation = selected;
  const data = selected ? await api(`/api/results/evaluation?suite=${suite}&experiment=${selected}`) : null;
  const balance = await loadBalance();
  if (renderVersion !== state.renderVersion) return;
  const metrics = data?.metrics?.overall || {};
  const confusion = data?.metrics?.confusion_matrix || {};
  const job = state.jobs.evaluation;
  app.innerHTML = `${heading("/evaluation", suiteSwitcher(suite, "evaluation"))}
    <section class="card card-pad" style="margin-bottom:16px"><div class="form-grid" style="align-items:end">
      <div class="field"><label class="field-label" for="evaluation-experiment">Model</label><select class="input" id="evaluation-experiment" ${experiments.length ? "" : "disabled"}>${experimentOptions(experiments, selected)}</select></div>
      <div class="field"><span class="field-label">Tập đánh giá</span><div class="input" style="display:flex;align-items:center;background:#faf9f6">Test · ${suite === "video" ? "pothole video frames" : "road damage images"}</div></div>
      <div class="span-two button-row"><button class="button primary" id="run-evaluation" ${experiments.length ? "" : "disabled"}>Chạy đánh giá test</button><span class="field-help">Tạo kết quả riêng, không ghi đè báo cáo hiện có.</span></div>
    </div><div id="evaluation-job">${evaluationJobMarkup(job)}</div></section>
    ${!data?.metrics ? `<div class="banner info" style="margin-bottom:16px"><span class="banner-icon">i</span><div><strong>Chưa có báo cáo cho lựa chọn này</strong><p>Sau khi hoàn tất đánh giá, tải lại trang để đọc metrics.json và ma trận nhầm lẫn.</p></div></div>` : ""}
    <div class="stats-grid" style="margin-bottom:16px">
      ${[["Precision", metrics.precision], ["Recall", metrics.recall], ["mAP50", metrics.mAP50], ["mAP50–95", metrics.mAP50_95]].map(([label, value]) => `<div class="stat-tile"><span class="stat-label">${label}</span><strong class="metric-value">${formatNumber(value)}</strong><span class="metric-unit">tập test · ${escapeHTML(selected || "—")}</span></div>`).join("")}
    </div>
    ${suite === "video" ? `<div class="banner" style="margin-bottom:16px"><span class="banner-icon">!</span><div><strong>Model video chỉ được học nhãn pothole</strong><p>Crack và manhole không có đối tượng ground truth trong bộ frame video.</p></div></div>` : ""}
    <div class="split-layout">
      ${card("Ma trận nhầm lẫn", escapeHTML(confusion.orientation || "Số lượng trên tập test."), `<div class="card-head" style="align-items:center"><span class="section-kicker">PREDICTED × ACTUAL</span><div class="segmented"><button type="button" data-matrix-mode="count" aria-pressed="${!state.matrixNormalized}">Số lượng</button><button type="button" data-matrix-mode="normalized" aria-pressed="${state.matrixNormalized}">Chuẩn hoá</button></div></div>${matrixMarkup(confusion, state.matrixNormalized)}`)}
      ${card("Theo lớp", "Precision, recall và average precision.", renderClassMetrics(metrics, data?.class_metrics))}
    </div>
    <div class="split-layout" style="margin-top:16px">
      ${card("Cân bằng lớp & kích thước box", "Báo cáo mô tả dataset ảnh hiện có; không tự thay đổi cách train.", balance)}
      ${card("Phân tích nhầm lẫn", "Các trường hợp cần theo dõi được đọc từ confusion_analysis.json.", renderConfusions(data?.confusion_analysis))}
    </div>
    <p class="footer-note">Evaluation dùng experiment_config.yaml đi cùng checkpoint để giữ đúng hợp đồng tiền xử lý. Kết quả từ GPU Colab và CPU cục bộ cần được ghi rõ thiết bị khi so độ trễ.</p>`;
  connectSuiteSwitcher("evaluation", (value) => { state.evaluationSuite = value; });
  document.getElementById("evaluation-experiment")?.addEventListener("change", (event) => {
    state.experiment.evaluation = event.target.value;
    render();
  });
  connectMatrixToggle();
  document.getElementById("run-evaluation")?.addEventListener("click", runEvaluation);
}

async function loadBalance() {
  const data = await api("/api/results/class-balance");
  return data.available ? renderBalance(data.report) : renderBalance(null);
}

function renderConfusions(data) {
  const entries = Object.entries(data?.important_confusions || {});
  if (!entries.length) return emptyState("Chưa có phân tích lỗi", "Không tìm thấy confusion_analysis.json.");
  return `<div class="stack">${entries.map(([key, item]) => `<div class="progress-head"><span>${escapeHTML(item.actual)} → ${escapeHTML(item.predicted)}</span><strong>${item.count ?? "—"}</strong></div>`).join("")}</div>`;
}

async function runEvaluation() {
  const button = document.getElementById("run-evaluation");
  const experiment = document.getElementById("evaluation-experiment")?.value;
  if (!experiment) return;
  button.disabled = true;
  const jobSlot = document.getElementById("evaluation-job");
  jobSlot.innerHTML = `<div class="job-status"><span class="spinner"></span><span>Đang đưa đánh giá vào hàng đợi…</span></div>`;
  try {
    const response = await api("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ suite: state.evaluationSuite, experiment }),
    });
    await watchEvaluation(response.job_id);
  } catch (error) {
    state.jobs.evaluation = { status: "failed", error: error.message };
    jobSlot.innerHTML = evaluationJobMarkup(state.jobs.evaluation);
  } finally {
    button.disabled = false;
  }
}

async function watchEvaluation(id) {
  state.jobs.evaluation = { status: "queued" };
  while (true) {
    const job = await api(`/api/jobs/${id}`);
    state.jobs.evaluation = job;
    if (state.route === "/evaluation") {
      const slot = document.getElementById("evaluation-job");
      if (slot) slot.innerHTML = evaluationJobMarkup(job);
    }
    if (!["queued", "running"].includes(job.status)) break;
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  if (state.route === "/evaluation" && state.jobs.evaluation.status === "completed") await render();
}

async function render() {
  state.route = routeFromHash();
  const renderVersion = ++state.renderVersion;
  updateNavigation();
  try {
    if (state.route === "/") await renderImagePage(renderVersion);
    else if (state.route === "/video") await renderVideoPage(renderVersion);
    else if (state.route === "/experiments") await renderExperimentsPage(renderVersion);
    else await renderEvaluationPage(renderVersion);
  } catch (error) {
    if (renderVersion !== state.renderVersion) return;
    app.innerHTML = `${heading(state.route)}<div class="banner danger"><span class="banner-icon">!</span><div><strong>Không tải được dữ liệu</strong><p>${escapeHTML(error.message)}</p><button class="button secondary small" type="button" onclick="location.reload()" style="margin-top:10px">Thử lại</button></div></div>`;
  }
}

window.addEventListener("hashchange", render);
render();
