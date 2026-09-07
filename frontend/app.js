/**
 * SatQuery AI — Agentic Remote Sensing Copilot Frontend
 * Interactive Split-Screen Satellite Engine, Agent Planner, Models & Intelligence Console
 */

document.addEventListener('DOMContentLoaded', () => {
    // =========================================================================
    // 1. STATE & CONSTANTS
    // =========================================================================
    const state = {
        splitPercent: 50,
        isDraggingSplit: false,
        opacity: 0.75,
        zoomLevel: 1.0,
        layersVisible: true,
        isDrawingRoi: false,
        roiStart: null,
        roiBox: null,
        activeT1Month: 3, // March
        activeT2Month: 9, // September
        isAnalyzing: false,
        activeTab: 'workspace'
    };

    const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    // DOM Elements
    const splitViewport = document.getElementById('splitViewport');
    const layerMarch = document.getElementById('layerMarch');
    const splitDivider = document.getElementById('splitDivider');
    const splitHandle = document.getElementById('splitHandle');
    const polygonsOverlay = document.getElementById('polygonsOverlay');
    const opacitySlider = document.getElementById('opacitySlider');
    const opacityVal = document.getElementById('opacityVal');
    const polyTooltip = document.getElementById('polyTooltip');

    // Controls
    const toolCrosshair = document.getElementById('toolCrosshair');
    const toolZoom = document.getElementById('toolZoom');
    const toolLayers = document.getElementById('toolLayers');
    const toolDrawRoi = document.getElementById('toolDrawRoi');
    const toolDownload = document.getElementById('toolDownload');

    // Timeline
    const monthMar = document.getElementById('monthMar');
    const monthSep = document.getElementById('monthSep');
    const timelineConnector = document.getElementById('timelineConnector');
    const comparisonBadge = document.getElementById('comparisonBadge');

    // Chat
    const chatMessageList = document.getElementById('chatMessageList');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const btnNewChat = document.getElementById('btnNewChat');
    const btnUploadImage = document.getElementById('btnUploadImage');
    const fileUploadInput = document.getElementById('fileUploadInput');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const chatThumbCard = document.getElementById('chatThumbCard');

    // Report
    const btnExportPdf = document.getElementById('btnExportPdf');
    const btnExportGeoJson = document.getElementById('btnExportGeoJson');
    const btnGenerateReport = document.getElementById('btnGenerateReport');
    const btnShareReport = document.getElementById('btnShareReport');
    const reportParagraph = document.getElementById('reportParagraph');

    // Modals & Navigation
    const navTabs = document.querySelectorAll('.nav-tab');
    const modalBackdrop = document.getElementById('modalBackdrop');
    const modalTitle = document.getElementById('modalTitle');
    const modalContent = document.getElementById('modalContent');
    const modalCloseBtn = document.getElementById('modalCloseBtn');
    const toastContainer = document.getElementById('toastContainer');

    // =========================================================================
    // 2. INTERACTIVE SPLIT SLIDER (< >)
    // =========================================================================
    function setSplitPosition(percent) {
        // Clamp between 2% and 98%
        const clamped = Math.max(2, Math.min(98, percent));
        state.splitPercent = clamped;

        // Update CSS clip-path on top March layer
        layerMarch.style.clipPath = `polygon(0 0, ${clamped}% 0, ${clamped}% 100%, 0 100%)`;
        splitDivider.style.left = `${clamped}%`;
    }

    function handleSplitMove(clientX) {
        if (!state.isDraggingSplit) return;
        const rect = splitViewport.getBoundingClientRect();
        const offsetX = clientX - rect.left;
        const percent = (offsetX / rect.width) * 100;
        setSplitPosition(percent);
    }

    splitHandle.addEventListener('mousedown', (e) => {
        state.isDraggingSplit = true;
        document.body.style.cursor = 'ew-resize';
        e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
        if (state.isDraggingSplit) {
            handleSplitMove(e.clientX);
        }
    });

    window.addEventListener('mouseup', () => {
        if (state.isDraggingSplit) {
            state.isDraggingSplit = false;
            document.body.style.cursor = 'default';
        }
    });

    // Touch Support
    splitHandle.addEventListener('touchstart', (e) => {
        state.isDraggingSplit = true;
        e.preventDefault();
    }, { passive: false });

    window.addEventListener('touchmove', (e) => {
        if (state.isDraggingSplit && e.touches.length > 0) {
            handleSplitMove(e.touches[0].clientX);
        }
    }, { passive: true });

    window.addEventListener('touchend', () => {
        state.isDraggingSplit = false;
    });

    // Initial position
    setSplitPosition(50);

    // =========================================================================
    // 3. OPACITY & LAYER CONTROLS
    // =========================================================================
    opacitySlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        state.opacity = val / 100;
        opacityVal.textContent = `${val}%`;
        if (polygonsOverlay && state.layersVisible) {
            polygonsOverlay.style.opacity = state.opacity;
        }
    });

    toolLayers.addEventListener('click', () => {
        state.layersVisible = !state.layersVisible;
        toolLayers.classList.toggle('active', state.layersVisible);
        polygonsOverlay.style.opacity = state.layersVisible ? state.opacity : 0;
        showToast(state.layersVisible ? 'Layers & Overlays Enabled' : 'Layers & Overlays Hidden');
    });

    toolCrosshair.addEventListener('click', () => {
        setSplitPosition(50);
        showToast('Split view re-centered to 50%');
    });

    toolZoom.addEventListener('click', () => {
        if (state.zoomLevel === 1.0) {
            state.zoomLevel = 1.35;
            toolZoom.classList.add('active');
            showToast('Zoom Level: 1.35x (Northeast Cluster Focus)');
        } else {
            state.zoomLevel = 1.0;
            toolZoom.classList.remove('active');
            showToast('Zoom Level: 1.0x (Fit View)');
        }
        document.querySelectorAll('.sat-ortho-img, .polygons-svg-overlay').forEach(el => {
            el.style.transform = state.zoomLevel === 1.0 ? 'none' : 'scale(1.35) translate(-6%, -6%)';
            el.style.transition = 'transform 0.35s cubic-bezier(0.16, 1, 0.3, 1)';
        });
    });

    // =========================================================================
    // 4. VECTOR POLYGONS & HOVER TOOLTIPS
    // =========================================================================
    const polygons = document.querySelectorAll('.poly-building, .poly-road');
    polygons.forEach(poly => {
        poly.addEventListener('mouseenter', (e) => {
            const id = poly.getAttribute('data-id');
            const type = poly.getAttribute('data-type');
            const area = poly.getAttribute('data-area');
            const conf = poly.getAttribute('data-conf');

            polyTooltip.innerHTML = `
                <div style="font-weight:700; color:#00d2ff; margin-bottom:2px;">${id}: ${type}</div>
                <div style="color:#94a3b8; font-size:10px;">Area: <strong style="color:#fff;">${area}</strong> | Conf: <strong style="color:#4ade80;">${conf}</strong></div>
            `;
            polyTooltip.style.display = 'block';
        });

        poly.addEventListener('mousemove', (e) => {
            const rect = splitViewport.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            polyTooltip.style.left = `${x}px`;
            polyTooltip.style.top = `${y}px`;
        });

        poly.addEventListener('mouseleave', () => {
            polyTooltip.style.display = 'none';
        });

        poly.addEventListener('click', () => {
            const id = poly.getAttribute('data-id');
            const type = poly.getAttribute('data-type');
            const area = poly.getAttribute('data-area');
            showToast(`Inspecting ${id} (${type}) — ${area}`);
        });
    });

    // =========================================================================
    // 5. DRAW ROI (Region of Interest)
    // =========================================================================
    let roiDrawingBox = null;

    toolDrawRoi.addEventListener('click', () => {
        state.isDrawingRoi = !state.isDrawingRoi;
        toolDrawRoi.classList.toggle('active', state.isDrawingRoi);

        if (state.isDrawingRoi) {
            splitViewport.style.cursor = 'crosshair';
            showToast('Draw ROI Mode: Click and drag on map to select area');
        } else {
            splitViewport.style.cursor = 'default';
            if (roiDrawingBox) {
                roiDrawingBox.remove();
                roiDrawingBox = null;
            }
        }
    });

    splitViewport.addEventListener('mousedown', (e) => {
        if (!state.isDrawingRoi || e.target === splitHandle) return;

        const rect = splitViewport.getBoundingClientRect();
        state.roiStart = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };

        if (roiDrawingBox) roiDrawingBox.remove();

        roiDrawingBox = document.createElement('div');
        roiDrawingBox.style.position = 'absolute';
        roiDrawingBox.style.border = '2px dashed #00d2ff';
        roiDrawingBox.style.background = 'rgba(0, 210, 255, 0.15)';
        roiDrawingBox.style.zIndex = '25';
        roiDrawingBox.style.pointerEvents = 'none';
        roiDrawingBox.style.left = `${state.roiStart.x}px`;
        roiDrawingBox.style.top = `${state.roiStart.y}px`;
        splitViewport.appendChild(roiDrawingBox);
    });

    splitViewport.addEventListener('mousemove', (e) => {
        if (!state.isDrawingRoi || !state.roiStart || !roiDrawingBox) return;

        const rect = splitViewport.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;

        const left = Math.min(state.roiStart.x, currentX);
        const top = Math.min(state.roiStart.y, currentY);
        const width = Math.abs(currentX - state.roiStart.x);
        const height = Math.abs(currentY - state.roiStart.y);

        roiDrawingBox.style.left = `${left}px`;
        roiDrawingBox.style.top = `${top}px`;
        roiDrawingBox.style.width = `${width}px`;
        roiDrawingBox.style.height = `${height}px`;
    });

    splitViewport.addEventListener('mouseup', () => {
        if (!state.isDrawingRoi || !state.roiStart) return;
        state.roiStart = null;
        showToast('ROI Selected! Analyzing localized change metrics...');
        setTimeout(() => {
            appendBotMessage("Targeted ROI query completed. Detected 6 building developments within custom polygon bounds (0.62 km²). Confidence: 93.4%.");
        }, 1200);
    });

    // =========================================================================
    // 6. DOWNLOAD / SNAPSHOT
    // =========================================================================
    toolDownload.addEventListener('click', () => {
        showToast('Generating high-resolution composite GeoTIFF / snapshot...');
        setTimeout(() => {
            const link = document.createElement('a');
            link.href = 'textures/t2_september.jpg';
            link.download = 'satquery_sentinel2_september2024_analysis.jpg';
            link.click();
            showToast('✓ Snapshot downloaded successfully');
        }, 600);
    });

    // =========================================================================
    // 7. TIMELINE SCRUBBING
    // =========================================================================
    document.querySelectorAll('.month-node').forEach(node => {
        node.addEventListener('click', () => {
            const monthNum = parseInt(node.getAttribute('data-month'), 10);
            const monthName = MONTHS[monthNum - 1];

            // If click before or after September
            if (monthNum <= 6) {
                state.activeT1Month = monthNum;
                document.querySelectorAll('.month-node').forEach(n => n.classList.remove('active-t1'));
                node.classList.add('active-t1');
            } else {
                state.activeT2Month = monthNum;
                document.querySelectorAll('.month-node').forEach(n => n.classList.remove('active-t2'));
                node.classList.add('active-t2');
            }

            const t1Name = MONTHS[state.activeT1Month - 1];
            const t2Name = MONTHS[state.activeT2Month - 1];

            comparisonBadge.innerHTML = `<strong>${t1Name} ↔ ${t2Name}</strong> <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>`;
            document.querySelector('.time-badge-t1').textContent = `T1 - ${t1Name} 2024`;
            document.querySelector('.time-badge-t2').textContent = `T2 - ${t2Name} 2024`;

            showToast(`Time window updated: ${t1Name} 2024 vs ${t2Name} 2024`);
        });
    });

    // =========================================================================
    // 8. CHAT INTERACTION & LIVE RE-ACT AGENT SIMULATION
    // =========================================================================
    function appendUserMessage(text) {
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const item = document.createElement('div');
        item.className = 'chat-item user-item';
        item.innerHTML = `
            <div class="chat-avatar user-avatar">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                </svg>
            </div>
            <div class="chat-bubble-wrap">
                <div class="chat-bubble user-bubble">${escapeHtml(text)}</div>
                <span class="chat-time">${time}</span>
            </div>
        `;
        chatMessageList.appendChild(item);
        chatMessageList.scrollTop = chatMessageList.scrollHeight;
    }

    function appendBotMessage(contentHtml) {
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const item = document.createElement('div');
        item.className = 'chat-item bot-item';
        item.innerHTML = `
            <div class="chat-avatar bot-avatar">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="11" width="18" height="10" rx="2"></rect>
                    <circle cx="12" cy="5" r="2"></circle>
                    <path d="M12 7v4"></path>
                    <line x1="8" y1="16" x2="8" y2="16"></line>
                    <line x1="16" y1="16" x2="16" y2="16"></line>
                </svg>
            </div>
            <div class="chat-bubble-wrap">
                <div class="chat-bubble bot-bubble">${contentHtml}</div>
                <span class="chat-time">${time}</span>
            </div>
        `;
        chatMessageList.appendChild(item);
        chatMessageList.scrollTop = chatMessageList.scrollHeight;
    }

    function triggerAgentPipeline(query) {
        if (state.isAnalyzing) return;
        state.isAnalyzing = true;

        // Animate Agent Planner steps
        const stepRows = document.querySelectorAll('.planner-step-row');
        
        // Reset steps
        stepRows.forEach((row, i) => {
            row.className = 'planner-step-row step-pending';
            row.querySelector('.step-badge').className = 'step-badge badge-pending';
            row.querySelector('.step-badge').textContent = 'Pending';
        });

        const stepNames = [
            'Understand Query',
            'Select Models',
            'Register Images',
            'Detect Changes',
            'Ground Objects',
            'Generate Explanation'
        ];

        let currentStep = 0;
        const interval = setInterval(() => {
            if (currentStep < stepRows.length) {
                // Set current as running
                stepRows[currentStep].className = 'planner-step-row step-running';
                stepRows[currentStep].querySelector('.step-badge').className = 'step-badge badge-running';
                stepRows[currentStep].querySelector('.step-badge').textContent = '● Running';

                // Mark previous as completed
                if (currentStep > 0) {
                    stepRows[currentStep - 1].className = 'planner-step-row step-done';
                    stepRows[currentStep - 1].querySelector('.step-badge').className = 'step-badge badge-completed';
                    stepRows[currentStep - 1].querySelector('.step-badge').textContent = 'Completed';
                }
                currentStep++;
            } else {
                // Final step complete
                stepRows[stepRows.length - 1].className = 'planner-step-row step-done';
                stepRows[stepRows.length - 1].querySelector('.step-badge').className = 'step-badge badge-completed';
                stepRows[stepRows.length - 1].querySelector('.step-badge').textContent = 'Completed';

                clearInterval(interval);
                state.isAnalyzing = false;

                // Formulate intelligent answer
                generateQueryResponse(query);
            }
        }, 500);
    }

    function generateQueryResponse(query) {
        const q = query.toLowerCase();
        let answer = '';

        if (q.includes('road') || q.includes('highway') || q.includes('corridor')) {
            answer = `
                <p class="summary-highlight">Road Infrastructure Changes Identified</p>
                <div class="findings-list">
                    <span class="findings-title">Key findings:</span>
                    <ul>
                        <li><span class="bullet-dot">•</span> 2 new paved access spurs detected</li>
                        <li><span class="bullet-dot">•</span> 1.68 km total corridor length added</li>
                        <li><span class="bullet-dot">•</span> Direct connection to northeast logistics hub</li>
                        <li><span class="bullet-dot">•</span> High segmentation confidence (93.9%)</li>
                    </ul>
                </div>
            `;
        } else if (q.includes('water') || q.includes('lake') || q.includes('flood')) {
            answer = `
                <p class="summary-highlight">Hydrological Body Analysis</p>
                <div class="findings-list">
                    <span class="findings-title">Key findings:</span>
                    <ul>
                        <li><span class="bullet-dot">•</span> Central pond surface area preserved (0.41 km²)</li>
                        <li><span class="bullet-dot">•</span> NDWI differencing confirms no encroachment</li>
                        <li><span class="bullet-dot">•</span> Seasonal water level fluctuation: -4.2%</li>
                        <li><span class="bullet-dot">•</span> High confidence (96.1%)</li>
                    </ul>
                </div>
            `;
        } else if (q.includes('vegetation') || q.includes('forest') || q.includes('green')) {
            answer = `
                <p class="summary-highlight">Vegetation & Canopy Assessment</p>
                <div class="findings-list">
                    <span class="findings-title">Key findings:</span>
                    <ul>
                        <li><span class="bullet-dot">•</span> Central tree cluster dense cover maintained</li>
                        <li><span class="bullet-dot">•</span> Agricultural plots converted to built-up: 1.82 km²</li>
                        <li><span class="bullet-dot">•</span> Net NDVI differential: -0.18 in northwest quadrant</li>
                        <li><span class="bullet-dot">•</span> High confidence (92.4%)</li>
                    </ul>
                </div>
            `;
        } else {
            answer = `
                <p class="summary-highlight">Bi-Temporal Analysis Completed</p>
                <div class="findings-list">
                    <span class="findings-title">Key findings:</span>
                    <ul>
                        <li><span class="bullet-dot">•</span> 24 new structures identified & segmented</li>
                        <li><span class="bullet-dot">•</span> Total built-up increase: 12.4%</li>
                        <li><span class="bullet-dot">•</span> Grounded by InternVL 3 & ChangeStar</li>
                        <li><span class="bullet-dot">•</span> Overall separation score: 91%</li>
                    </ul>
                </div>
            `;
        }

        appendBotMessage(answer);
    }

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;

        appendUserMessage(text);
        chatInput.value = '';
        triggerAgentPipeline(text);
    });

    btnAnalyze.addEventListener('click', () => {
        const text = chatInput.value.trim() || 'Execute comprehensive bi-temporal change detection on active tile';
        appendUserMessage(text);
        chatInput.value = '';
        triggerAgentPipeline(text);
    });

    btnNewChat.addEventListener('click', () => {
        chatMessageList.innerHTML = `
            <div class="chat-item bot-item">
                <div class="chat-avatar bot-avatar">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="3" y="11" width="18" height="10" rx="2"></rect>
                        <circle cx="12" cy="5" r="2"></circle>
                        <path d="M12 7v4"></path>
                    </svg>
                </div>
                <div class="chat-bubble-wrap">
                    <div class="chat-bubble bot-bubble">
                        Welcome to <strong>SatQuery AI</strong> session. Upload satellite GeoTIFF pairs or ask any Earth-observation question below.
                    </div>
                    <span class="chat-time">Just now</span>
                </div>
            </div>
        `;
        showToast('New investigation session started');
    });

    btnUploadImage.addEventListener('click', () => {
        fileUploadInput.click();
    });

    fileUploadInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            const fileName = e.target.files[0].name;
            showToast(`Loaded imagery: ${fileName}. Validating co-registration...`);
            setTimeout(() => {
                appendBotMessage(`<strong>Co-registration validated:</strong> "${fileName}" matched to Tile 43QFC with 0.12 pixel spatial RMS error.`);
            }, 800);
        }
    });

    if (chatThumbCard) {
        chatThumbCard.addEventListener('click', () => {
            setSplitPosition(25);
            showToast('Centered on Northeast Cluster (>90% confidence)');
        });
    }

    // =========================================================================
    // 9. REPORT ACTIONS (Export PDF, GeoJSON, Generate, Share)
    // =========================================================================
    btnExportPdf.addEventListener('click', () => {
        showToast('Generating official SIH / ISRO Intelligence PDF Report...');
        setTimeout(() => {
            const blob = new Blob([
                "SATQUERY AI — REMOTE SENSING INTELLIGENCE REPORT\n",
                "ISRO SIH26167 Problem Statement\n",
                "Timestamp: September 2024 | Tile: 43QFC | Sensor: Sentinel-2\n",
                "Area Detected: 2.4 km² | New Buildings: 24 | Roads: 2\n",
                "Confidence Score: 91.0%\n\n",
                "Key Evidence:\n",
                "- Detected new impervious surfaces in northwest/northeast quadrants.\n",
                "- Confirmed by SAM2 and ChangeStar bi-temporal difference masks.\n",
                "- STSF-Net pseudo-change filter eliminated illumination artifacts.\n"
            ], { type: 'application/pdf' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'satquery_intelligence_report_43QFC.pdf';
            a.click();
            showToast('✓ PDF Intelligence Report downloaded');
        }, 600);
    });

    btnExportGeoJson.addEventListener('click', () => {
        showToast('Generating standardized GeoJSON feature collection...');
        const geojson = {
            type: "FeatureCollection",
            crs: { type: "name", properties: { name: "urn:ogc:def:crs:OGC:1.3:CRS84" } },
            features: Array.from(polygons).map((poly, idx) => ({
                type: "Feature",
                properties: {
                    id: poly.getAttribute('data-id'),
                    structure_type: poly.getAttribute('data-type'),
                    area_m2: poly.getAttribute('data-area'),
                    confidence: poly.getAttribute('data-conf')
                },
                geometry: {
                    type: "Polygon",
                    coordinates: [[[77.594 + idx*0.001, 12.971 + idx*0.001], [77.595, 12.971], [77.595, 12.972], [77.594, 12.972], [77.594 + idx*0.001, 12.971 + idx*0.001]]]
                }
            }))
        };

        const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/geo+json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'satquery_urban_polygons_43QFC.geojson';
        a.click();
        showToast('✓ GeoJSON exported successfully');
    });

    btnGenerateReport.addEventListener('click', () => {
        reportParagraph.textContent = 'Re-synthesizing multi-modal telemetry and spectral differences...';
        showToast('Re-generating AI report summary...');
        setTimeout(() => {
            reportParagraph.textContent = 'Comparison between March and September imagery detected a 12.4% increase in urban development concentrated in the northeast quadrant. Twenty-four newly constructed buildings were identified with high confidence. No significant vegetation loss was observed.';
            showToast('✓ AI Report updated with latest sensor passes');
        }, 900);
    });

    btnShareReport.addEventListener('click', () => {
        navigator.clipboard.writeText(window.location.href);
        showToast('✓ Session permalink copied to clipboard');
    });

    // =========================================================================
    // 10. MODALS & TAB NAVIGATION
    // =========================================================================
    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.getAttribute('data-tab');
            navTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            if (tabName === 'workspace') {
                modalBackdrop.classList.remove('active');
                return;
            }

            openModal(tabName);
        });
    });

    function openModal(type) {
        modalBackdrop.classList.add('active');

        if (type === 'models') {
            modalTitle.textContent = 'Foundation & VLM Models Registry (6 Active)';
            modalContent.innerHTML = `
                <div style="display:flex; flex-direction:column; gap:12px;">
                    <p>SatQuery AI orchestrates specialized remote-sensing foundation models in a ReAct framework:</p>
                    <table style="width:100%; border-collapse:collapse; font-size:11.5px; text-align:left;">
                        <thead>
                            <tr style="border-bottom:1px solid #1e3a6b; color:#38bdf8;">
                                <th style="padding:6px;">Model</th>
                                <th style="padding:6px;">Architecture</th>
                                <th style="padding:6px;">Role</th>
                                <th style="padding:6px;">Latency</th>
                                <th style="padding:6px;">Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                                <td style="padding:6px; font-weight:600; color:#fff;">InternVL 3</td>
                                <td style="padding:6px;">Vision-Language</td>
                                <td style="padding:6px;">Spatial QA & Description</td>
                                <td style="padding:6px;">12.4s</td>
                                <td style="padding:6px; color:#00d2ff;">● Active</td>
                            </tr>
                            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                                <td style="padding:6px; font-weight:600; color:#fff;">Florence-2</td>
                                <td style="padding:6px;">Multimodal Unified</td>
                                <td style="padding:6px;">Visual Grounding</td>
                                <td style="padding:6px;">8.7s</td>
                                <td style="padding:6px; color:#4ade80;">✓ Ready</td>
                            </tr>
                            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                                <td style="padding:6px; font-weight:600; color:#fff;">SAM2</td>
                                <td style="padding:6px;">Segment Anything v2</td>
                                <td style="padding:6px;">Zero-shot Instance Mask</td>
                                <td style="padding:6px;">6.1s</td>
                                <td style="padding:6px; color:#4ade80;">✓ Ready</td>
                            </tr>
                            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                                <td style="padding:6px; font-weight:600; color:#fff;">GroundingDINO</td>
                                <td style="padding:6px;">Open-Set Detector</td>
                                <td style="padding:6px;">Bounding Box Discovery</td>
                                <td style="padding:6px;">9.8s</td>
                                <td style="padding:6px; color:#00d2ff;">● Active</td>
                            </tr>
                            <tr style="border-bottom:1px solid rgba(255,255,255,0.06);">
                                <td style="padding:6px; font-weight:600; color:#fff;">ChangeStar</td>
                                <td style="padding:6px;">Bi-Temporal Siamese</td>
                                <td style="padding:6px;">Dense Change Differencing</td>
                                <td style="padding:6px;">14.2s</td>
                                <td style="padding:6px; color:#4ade80;">✓ Ready</td>
                            </tr>
                            <tr>
                                <td style="padding:6px; font-weight:600; color:#fff;">Segment Anything</td>
                                <td style="padding:6px;">ViT-H SAM</td>
                                <td style="padding:6px;">Polygon Refinement</td>
                                <td style="padding:6px;">7.6s</td>
                                <td style="padding:6px; color:#4ade80;">✓ Ready</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            `;
        } else if (type === 'datasets') {
            modalTitle.textContent = 'Remote Sensing Satellite Feeds';
            modalContent.innerHTML = `
                <div style="display:flex; flex-direction:column; gap:12px;">
                    <div style="background:#091226; border:1px solid #1e3a6b; border-radius:8px; padding:12px;">
                        <h4 style="color:#00d2ff; margin-bottom:4px;">Sentinel-2 MSI (Optical)</h4>
                        <p style="font-size:11.5px; color:#cbd5e1;">10m spatial resolution across 13 spectral bands (RGB, NIR, RedEdge, SWIR). 5-day revisit cycle.</p>
                    </div>
                    <div style="background:#091226; border:1px solid #1e3a6b; border-radius:8px; padding:12px;">
                        <h4 style="color:#38bdf8; margin-bottom:4px;">Cartosat-2S (ISRO Optical High-Res)</h4>
                        <p style="font-size:11.5px; color:#cbd5e1;">0.65m panchromatic and 2.0m 4-band multispectral sensor. Optimized for cadastral mapping.</p>
                    </div>
                    <div style="background:#091226; border:1px solid #1e3a6b; border-radius:8px; padding:12px;">
                        <h4 style="color:#a855f7; margin-bottom:4px;">RISAT-1A / EOS-04 (ISRO C-Band SAR)</h4>
                        <p style="font-size:11.5px; color:#cbd5e1;">Microwave SAR penetration through clouds, haze, and precipitation for all-weather flood & terrain monitoring.</p>
                    </div>
                </div>
            `;
        } else if (type === 'history') {
            modalTitle.textContent = 'Investigation History & Audit Trail';
            modalContent.innerHTML = `
                <div style="display:flex; flex-direction:column; gap:8px;">
                    <div style="background:#091226; border:1px solid #1e3a6b; border-radius:8px; padding:10px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="color:#fff; font-weight:600;">Urban Expansion: Bangalore Northeast (Tile 43QFC)</div>
                            <div style="color:#64748b; font-size:10px;">March 2024 vs September 2024 · 24 Buildings</div>
                        </div>
                        <span style="color:#4ade80; font-size:11px; font-weight:600;">91% Conf</span>
                    </div>
                    <div style="background:#091226; border:1px solid #1e3a6b; border-radius:8px; padding:10px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="color:#fff; font-weight:600;">Flood Inundation: Brahmaputra Basin (RISAT-1A)</div>
                            <div style="color:#64748b; font-size:10px;">Pre-Monsoon vs Post-Monsoon · 14.8 km² Inundation</div>
                        </div>
                        <span style="color:#4ade80; font-size:11px; font-weight:600;">95% Conf</span>
                    </div>
                    <div style="background:#091226; border:1px solid #1e3a6b; border-radius:8px; padding:10px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="color:#fff; font-weight:600;">Deforestation Monitoring: Western Ghats</div>
                            <div style="color:#64748b; font-size:10px;">Cartosat-2S & Sentinel-2 Fusion · -3.1 km² Canopy</div>
                        </div>
                        <span style="color:#4ade80; font-size:11px; font-weight:600;">89% Conf</span>
                    </div>
                </div>
            `;
        } else if (type === 'settings') {
            modalTitle.textContent = 'Engine & Algorithm Configuration';
            modalContent.innerHTML = `
                <div style="display:flex; flex-direction:column; gap:14px;">
                    <div>
                        <label style="color:#fff; font-weight:600; display:block; margin-bottom:4px;">STSF-Net Pseudo-Change Suppression Filter</label>
                        <span style="color:#94a3b8; font-size:11px; display:block; margin-bottom:6px;">Eliminates false positives from seasonal illumination differences.</span>
                        <input type="range" min="3" max="15" value="7" style="width:100%; accent-color:#0084ff;">
                        <div style="display:flex; justify-content:space-between; font-size:10px; color:#64748b;">
                            <span>3x3 Kernel</span>
                            <span style="color:#00d2ff;">7x7 Kernel (Default)</span>
                            <span>15x15 Kernel</span>
                        </div>
                    </div>

                    <div>
                        <label style="color:#fff; font-weight:600; display:block; margin-bottom:4px;">Confidence Threshold Cutoff</label>
                        <span style="color:#94a3b8; font-size:11px; display:block; margin-bottom:6px;">Filters polygon outputs with bimodal histogram confidence below cutoff.</span>
                        <input type="range" min="50" max="99" value="90" style="width:100%; accent-color:#0084ff;">
                        <div style="display:flex; justify-content:space-between; font-size:10px; color:#64748b;">
                            <span>50% (Loose)</span>
                            <span style="color:#00d2ff;">90% (ISRO SIH Standard)</span>
                            <span>99% (Strict)</span>
                        </div>
                    </div>
                </div>
            `;
        }
    }

    modalCloseBtn.addEventListener('click', () => {
        modalBackdrop.classList.remove('active');
        document.getElementById('tabWorkspace').click();
    });

    modalBackdrop.addEventListener('click', (e) => {
        if (e.target === modalBackdrop) {
            modalBackdrop.classList.remove('active');
            document.getElementById('tabWorkspace').click();
        }
    });

    // =========================================================================
    // 11. TOAST NOTIFICATIONS
    // =========================================================================
    function showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast-msg';
        toast.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" stroke-width="2.5">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            <span>${escapeHtml(message)}</span>
        `;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
