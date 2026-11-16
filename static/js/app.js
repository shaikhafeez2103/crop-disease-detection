/**
 * static/js/app.js — AgroScan AI (CropGuard ML) Frontend Controller
 *
 * Handles:
 * 1. Drag-and-drop & file input validation (PNG/JPG/JPEG/WEBP/BMP <= 10MB)
 * 2. Instant high-resolution leaf preview + "Remove / Change Image"
 * 3. One-click "Quick Test Sample Leaves" strip (`sample_images/`)
 * 4. Asynchronous POST `/predict` inference with animated scanning overlay
 * 5. Dynamic rendering of Confidence Bar, Top-3 Softmax Breakdown, Severity Badge,
 *    Tabbed Symptoms & Causes / Recommended Treatment, and Disclaimer Banner
 * 6. Session-based Recent Scan History strip with click-to-restore capability
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements — Upload & Samples
    const dropzone = document.getElementById("dropzone");
    const leafFileInput = document.getElementById("leafFileInput");
    const dropzonePrompt = document.getElementById("dropzonePrompt");
    const previewContainer = document.getElementById("previewContainer");
    const leafPreviewImg = document.getElementById("leafPreviewImg");
    const previewFilenameTag = document.getElementById("previewFilenameTag");
    const scanLaserLine = document.getElementById("scanLaserLine");
    const removeImageBtn = document.getElementById("removeImageBtn");
    const runDiagnosisBtn = document.getElementById("runDiagnosisBtn");
    const uploadErrorBanner = document.getElementById("uploadErrorBanner");
    const uploadErrorText = document.getElementById("uploadErrorText");
    const sampleThumbBtns = document.querySelectorAll(".sample-thumb-btn");

    // DOM Elements — Right Diagnostic Dashboard States
    const emptyStateView = document.getElementById("emptyStateView");
    const loadingStateView = document.getElementById("loadingStateView");
    const resultStateView = document.getElementById("resultStateView");

    // DOM Elements — Diagnosis Result Fields
    const resultHealthBadge = document.getElementById("resultHealthBadge");
    const resultSeverityBadge = document.getElementById("resultSeverityBadge");
    const resultLatencyTag = document.getElementById("resultLatencyTag");
    const resultDiseaseTitle = document.getElementById("resultDiseaseTitle");
    const resultScientificName = document.getElementById("resultScientificName");
    const resultCropHosts = document.getElementById("resultCropHosts");
    const resultConfidenceValue = document.getElementById("resultConfidenceValue");
    const resultConfidenceFill = document.getElementById("resultConfidenceFill");
    const top3BarsContainer = document.getElementById("top3BarsContainer");

    // DOM Elements — Tabs & Advisory Lists
    const tabBtns = document.querySelectorAll(".advisory-tab-btn");
    const treatmentTabPanel = document.getElementById("treatmentTabPanel");
    const symptomsTabPanel = document.getElementById("symptomsTabPanel");
    const treatmentSummaryText = document.getElementById("treatmentSummaryText");
    const organicTreatmentList = document.getElementById("organicTreatmentList");
    const chemicalTreatmentList = document.getElementById("chemicalTreatmentList");
    const symptomsSummaryText = document.getElementById("symptomsSummaryText");
    const symptomsBulletList = document.getElementById("symptomsBulletList");
    const causesBulletList = document.getElementById("causesBulletList");
    const resultDisclaimerText = document.getElementById("resultDisclaimerText");

    // DOM Elements — Session Scan History & Modal
    const scanHistoryStrip = document.getElementById("scanHistoryStrip");
    const historyEmptyPlaceholder = document.getElementById("historyEmptyPlaceholder");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    const openHowItWorksBtn = document.getElementById("openHowItWorksBtn");
    const closeHowItWorksBtn = document.getElementById("closeHowItWorksBtn");
    const howItWorksModal = document.getElementById("howItWorksModal");

    // State
    let selectedFile = null;
    let selectedSampleName = null;
    let currentPreviewDataUrl = "";
    let sessionHistory = [];

    const ALLOWED_MIME_TYPES = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp"
    ];
    const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB

    // =========================================================================
    // 1. Error & State Helpers
    // =========================================================================
    function showUploadError(message) {
        uploadErrorText.textContent = message;
        uploadErrorBanner.classList.remove("hidden");
    }

    function clearUploadError() {
        uploadErrorBanner.classList.add("hidden");
        uploadErrorText.textContent = "";
    }

    function switchDashboardState(state) {
        emptyStateView.classList.add("hidden");
        loadingStateView.classList.add("hidden");
        resultStateView.classList.add("hidden");

        if (state === "empty") {
            emptyStateView.classList.remove("hidden");
        } else if (state === "loading") {
            loadingStateView.classList.remove("hidden");
        } else if (state === "result") {
            resultStateView.classList.remove("hidden");
        }
    }

    // =========================================================================
    // 2. File Selection & Preview Handling
    // =========================================================================
    function validateAndPreviewFile(file) {
        clearUploadError();
        if (!file) return;

        const extValid = /\.(jpe?g|png|webp|bmp)$/i.test(file.name);
        if (!ALLOWED_MIME_TYPES.includes(file.type) && !extValid) {
            showUploadError("Unsupported file format. Please upload a PNG, JPG, or WEBP leaf photo.");
            return;
        }

        if (file.size > MAX_FILE_SIZE_BYTES) {
            showUploadError("File exceeds 10 MB limit. Please choose a smaller leaf image.");
            return;
        }

        selectedFile = file;
        selectedSampleName = null;
        sampleThumbBtns.forEach((btn) => btn.classList.remove("active"));

        const reader = new FileReader();
        reader.onload = (e) => {
            currentPreviewDataUrl = e.target.result;
            leafPreviewImg.src = currentPreviewDataUrl;
            previewFilenameTag.textContent = `${file.name} • 224×224 Ready`;
            dropzonePrompt.classList.add("hidden");
            previewContainer.classList.remove("hidden");
            // Automatically run inference upon upload for seamless UX
            runPredictionRequest();
        };
        reader.readAsDataURL(file);
    }

    function resetSelection() {
        clearUploadError();
        selectedFile = null;
        selectedSampleName = null;
        currentPreviewDataUrl = "";
        leafFileInput.value = "";
        leafPreviewImg.src = "";
        previewContainer.classList.add("hidden");
        dropzonePrompt.classList.remove("hidden");
        scanLaserLine.classList.add("hidden");
        sampleThumbBtns.forEach((btn) => btn.classList.remove("active"));
        switchDashboardState("empty");
    }

    // Dropzone click & keyboard events
    dropzone.addEventListener("click", (e) => {
        if (
            e.target.closest("#removeImageBtn") ||
            e.target.closest("#runDiagnosisBtn")
        ) {
            return;
        }
        if (previewContainer.classList.contains("hidden")) {
            leafFileInput.click();
        }
    });

    dropzone.addEventListener("keydown", (e) => {
        if ((e.key === "Enter" || e.key === " ") && previewContainer.classList.contains("hidden")) {
            e.preventDefault();
            leafFileInput.click();
        }
    });

    leafFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            validateAndPreviewFile(e.target.files[0]);
        }
    });

    // Drag and Drop events
    ["dragenter", "dragover"].forEach((evtName) => {
        dropzone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach((evtName) => {
        dropzone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("drag-over");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files[0]) {
            validateAndPreviewFile(dt.files[0]);
        }
    });

    removeImageBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetSelection();
    });

    runDiagnosisBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        runPredictionRequest();
    });

    // =========================================================================
    // 3. Quick Test Sample Leaves Strip
    // =========================================================================
    sampleThumbBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            clearUploadError();
            const sampleFile = btn.getAttribute("data-sample");
            const sampleLabel = btn.getAttribute("data-label");
            if (!sampleFile) return;

            sampleThumbBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            selectedFile = null;
            selectedSampleName = sampleFile;
            currentPreviewDataUrl = `/sample_images/${sampleFile}`;

            leafPreviewImg.src = currentPreviewDataUrl;
            previewFilenameTag.textContent = `${sampleLabel} (${sampleFile}) • 224×224`;
            dropzonePrompt.classList.add("hidden");
            previewContainer.classList.remove("hidden");

            runPredictionRequest();
        });
    });

    // =========================================================================
    // 4. API Call to `/predict` & Rendering Results
    // =========================================================================
    async function runPredictionRequest() {
        if (!selectedFile && !selectedSampleName) {
            showUploadError("Please select or upload a leaf image first.");
            return;
        }

        clearUploadError();
        switchDashboardState("loading");
        scanLaserLine.classList.remove("hidden");

        const formData = new FormData();
        if (selectedFile) {
            formData.append("file", selectedFile);
        } else if (selectedSampleName) {
            formData.append("sample_name", selectedSampleName);
        }

        try {
            const [response] = await Promise.all([
                fetch("/predict", {
                    method: "POST",
                    body: formData,
                }),
                // Brief minimum scan animation for visual polish
                new Promise((resolve) => setTimeout(resolve, 340)),
            ]);

            const data = await response.json();
            scanLaserLine.classList.add("hidden");

            if (!response.ok || data.error) {
                switchDashboardState("empty");
                showUploadError(data.error || "Prediction request failed.");
                return;
            }

            renderDiagnosisResult(data);
            recordScanInHistory(data, currentPreviewDataUrl);
        } catch (err) {
            scanLaserLine.classList.add("hidden");
            switchDashboardState("empty");
            showUploadError("Network error communicating with /predict endpoint.");
        }
    }

    function renderDiagnosisResult(data) {
        switchDashboardState("result");

        const isHealthy = data.predicted_class === "Healthy";
        const severity = data.severity_estimate || (isHealthy ? "Healthy" : "Moderate");

        // 1. Health Status Pill
        resultHealthBadge.className = "status-pill-badge";
        if (isHealthy) {
            resultHealthBadge.textContent = "✓ Healthy Foliage";
            resultHealthBadge.classList.add("badge-healthy");
        } else {
            resultHealthBadge.textContent = "⚠ Pathogen Detected";
            resultHealthBadge.classList.add(severity === "High" ? "badge-danger" : "badge-warning");
        }

        // 2. Severity Pill
        resultSeverityBadge.className = "severity-pill-badge";
        resultSeverityBadge.textContent = `Severity: ${severity}`;
        if (severity === "Healthy" || severity === "Low") {
            resultSeverityBadge.classList.add("badge-healthy");
        } else if (severity === "Moderate") {
            resultSeverityBadge.classList.add("badge-warning");
        } else {
            resultSeverityBadge.classList.add("badge-danger");
        }

        // 3. Latency & Titles
        resultLatencyTag.textContent = `⚡ ${data.inference_time_ms || 12.4} ms • MobileNetV2`;
        resultDiseaseTitle.textContent = data.predicted_class;
        resultScientificName.textContent = data.scientific_name || "PlantVillage Classification";
        resultCropHosts.textContent = data.crop_hosts ? `Hosts: ${data.crop_hosts}` : "";

        // 4. Primary Confidence Progress Bar
        const confScore = Number(data.confidence_score || 0).toFixed(2);
        resultConfidenceValue.textContent = `${confScore}%`;
        resultConfidenceFill.style.width = "0%";
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                resultConfidenceFill.style.width = `${Math.min(100, Math.max(0, confScore))}%`;
            });
        });

        // 5. Top-3 Softmax Breakdown Bars
        top3BarsContainer.innerHTML = "";
        const top3 = Array.isArray(data.top_3_predictions) ? data.top_3_predictions : [];
        top3.forEach((item) => {
            const pct = Number(item.confidence || 0).toFixed(2);
            const row = document.createElement("div");
            row.className = "top3-row";
            row.innerHTML = `
                <span class="top3-label" title="${item.label}">${item.label}</span>
                <div class="top3-track">
                    <div class="top3-fill" style="width: ${Math.min(100, Math.max(1, pct))}%;"></div>
                </div>
                <span class="top3-percent">${pct}%</span>
            `;
            top3BarsContainer.appendChild(row);
        });

        // 6. Recommended Treatment (Organic & Chemical)
        const treatment = data.recommended_treatment || {};
        treatmentSummaryText.textContent = treatment.summary || "";
        populateBulletList(organicTreatmentList, treatment.organic || []);
        populateBulletList(chemicalTreatmentList, treatment.chemical || []);

        // 7. Symptoms & Causes
        const symCauses = data.symptoms_and_causes || {};
        symptomsSummaryText.textContent = symCauses.summary || "";
        populateBulletList(symptomsBulletList, symCauses.symptoms || []);
        populateBulletList(causesBulletList, symCauses.causes || []);

        // 8. Disclaimer
        resultDisclaimerText.textContent =
            data.disclaimer ||
            "Estimate only — validate with an agricultural expert before applying chemical treatments.";
    }

    function populateBulletList(ulElement, items) {
        ulElement.innerHTML = "";
        items.forEach((text) => {
            const li = document.createElement("li");
            li.textContent = text;
            ulElement.appendChild(li);
        });
    }

    // =========================================================================
    // 5. Tab Switching (Recommended Treatment vs Symptoms & Causes)
    // =========================================================================
    tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            tabBtns.forEach((b) => {
                b.classList.remove("active");
                b.setAttribute("aria-selected", "false");
            });
            btn.classList.add("active");
            btn.setAttribute("aria-selected", "true");

            if (targetId === "treatmentTabPanel") {
                treatmentTabPanel.classList.remove("hidden");
                symptomsTabPanel.classList.add("hidden");
            } else {
                symptomsTabPanel.classList.remove("hidden");
                treatmentTabPanel.classList.add("hidden");
            }
        });
    });

    // =========================================================================
    // 6. Recent Scan History Strip (Session-Based)
    // =========================================================================
    function recordScanInHistory(resultData, thumbUrl) {
        const timestamp = new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        });

        const entry = {
            id: Date.now(),
            data: resultData,
            thumbUrl: thumbUrl,
            timestamp: timestamp,
        };

        sessionHistory.unshift(entry);
        if (sessionHistory.length > 10) {
            sessionHistory.pop();
        }
        renderHistoryStrip();
    }

    function renderHistoryStrip() {
        scanHistoryStrip.innerHTML = "";
        if (sessionHistory.length === 0) {
            scanHistoryStrip.appendChild(historyEmptyPlaceholder);
            historyEmptyPlaceholder.classList.remove("hidden");
            clearHistoryBtn.classList.add("hidden");
            return;
        }

        clearHistoryBtn.classList.remove("hidden");

        sessionHistory.forEach((entry) => {
            const card = document.createElement("div");
            card.className = "history-item-card";
            card.setAttribute("role", "button");
            card.setAttribute("tabindex", "0");
            card.innerHTML = `
                <img src="${entry.thumbUrl}" alt="${entry.data.predicted_class}" />
                <div class="history-item-meta">
                    <span class="history-item-title">${entry.data.predicted_class}</span>
                    <span class="history-item-conf">${Number(entry.data.confidence_score).toFixed(1)}% • ${entry.data.severity_estimate}</span>
                    <span class="history-item-time">Scanned at ${entry.timestamp}</span>
                </div>
            `;
            card.addEventListener("click", () => {
                leafPreviewImg.src = entry.thumbUrl;
                dropzonePrompt.classList.add("hidden");
                previewContainer.classList.remove("hidden");
                renderDiagnosisResult(entry.data);
            });
            scanHistoryStrip.appendChild(card);
        });
    }

    clearHistoryBtn.addEventListener("click", () => {
        sessionHistory = [];
        renderHistoryStrip();
    });

    // =========================================================================
    // 7. "How It Works" Modal Controls
    // =========================================================================
    openHowItWorksBtn.addEventListener("click", () => {
        howItWorksModal.classList.remove("hidden");
    });

    closeHowItWorksBtn.addEventListener("click", () => {
        howItWorksModal.classList.add("hidden");
    });

    howItWorksModal.addEventListener("click", (e) => {
        if (e.target === howItWorksModal) {
            howItWorksModal.classList.add("hidden");
        }
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !howItWorksModal.classList.contains("hidden")) {
            howItWorksModal.classList.add("hidden");
        }
    });
});
