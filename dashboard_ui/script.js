document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = 'http://localhost:8000';

    // --- Preloader Removal ---
    const preloader = document.getElementById('preloader');
    const hidePreloader = () => {
        if (preloader) {
            preloader.style.opacity = '0';
            setTimeout(() => preloader.style.display = 'none', 800);
        }
    };

    const safetyTimer = setTimeout(hidePreloader, 3000);
    window.addEventListener('load', () => {
        clearTimeout(safetyTimer);
        hidePreloader();
    });

    // --- State/Region Coordinates ---
    const REGIONS_DATA = {
        "Andhra Pradesh": { lat: 15.9, lon: 80.5, zoom: 7 },
        "Gujarat": { lat: 22.3, lon: 69.7, zoom: 7 },
        "Kerala": { lat: 9.5, lon: 76.3, zoom: 8 },
        "Maharashtra": { lat: 17.5, lon: 73.2, zoom: 7 },
        "Odisha Coast": { lat: 20.5, lon: 85.8, zoom: 7 },
        "Tamil Nadu": { lat: 10.8, lon: 79.8, zoom: 7 },
        "West Bengal": { lat: 21.6, lon: 87.9, zoom: 8 },
        "All": { lat: 18, lon: 82, zoom: 5 }
    };

    // --- Global State ---
    let latestPredictions = [];
    let latestReadings = [];
    let latestAlerts = [];
    let currentRegion = "All";
    let currentSeverityFilter = "all";

    // --- Map Initialization ---
    const lightTiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; CARTO'
    });
    const satelliteTiles = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Esri'
    });

    let map = L.map('map', {
        center: [18, 82],
        zoom: 5,
        layers: [lightTiles],
        zoomControl: false,
        attributionControl: false
    });

    L.control.zoom({ position: 'topright' }).addTo(map);
    let markersLayer = L.layerGroup().addTo(map);

    // Map Mode Toggles
    document.getElementById('mapSurvey').addEventListener('click', () => {
        map.removeLayer(satelliteTiles);
        map.addLayer(lightTiles);
        document.getElementById('mapSurvey').classList.add('active');
        document.getElementById('mapSat').classList.remove('active');
    });

    document.getElementById('mapSat').addEventListener('click', () => {
        map.removeLayer(lightTiles);
        map.addLayer(satelliteTiles);
        document.getElementById('mapSat').classList.add('active');
        document.getElementById('mapSurvey').classList.remove('active');
    });

    // --- Core Sync Engine ---
    const navLinks = document.querySelectorAll('.nav-links li');
    const tabContents = document.querySelectorAll('.tab-content');
    const globalSelector = document.getElementById('stateSelect');

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            const targetTab = link.getAttribute('data-tab');
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === targetTab) content.classList.add('active');
            });
            syncAllViews(currentRegion);
            if (targetTab === 'risk') setTimeout(() => map.invalidateSize(), 400);
        });
    });

    globalSelector.addEventListener('change', (e) => {
        currentRegion = e.target.value;
        syncAllViews(currentRegion);
    });

    function syncAllViews(region) {
        const activeTab = document.querySelector('.nav-links li.active').getAttribute('data-tab');
        if (activeTab === 'risk') updateRiskOverviewUI(region);
        else if (activeTab === 'analytics') loadAnalyticsData();
        else if (activeTab === 'broadcast') loadBroadcasts();
    }

    // --- Number Animation Logic ---
    function animateValue(el, start, end, duration) {
        if (!el) return;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const currentVal = (progress * (end - start)) + start;
            el.innerHTML = el.classList.contains('counter') && el.innerHTML.includes('%') ?
                          `${currentVal.toFixed(1)}%` : Math.floor(currentVal).toString().padStart(2, '0');
            if (progress < 1) window.requestAnimationFrame(step);
        };
        window.requestAnimationFrame(step);
    }

    // --- Operational Dashboard Logic ---
    async function initSystem() {
        try {
            const [preds, reads, alerts] = await Promise.all([
                fetch(`${API_BASE}/predictions`).then(res => res.json()),
                fetch(`${API_BASE}/readings`).then(res => res.json()),
                fetch(`${API_BASE}/alerts`).then(res => res.json())
            ]);
            latestPredictions = preds;
            latestReadings = reads;
            latestAlerts = alerts;

            if (latestPredictions.length > 0) {
                renderNationalOverview();
                plotGeospatialRisk();
                syncAllViews(currentRegion);
            }
        } catch (e) { console.error('System Offline:', e); }
    }

    function renderNationalOverview() {
        const top = [...latestPredictions].sort((a, b) => b.final_probability - a.final_probability)[0];
        const avg = (latestPredictions.reduce((acc, p) => acc + p.final_probability, 0) / latestPredictions.length) * 100;

        const kpis = document.querySelectorAll('.kpi-value');
        animateValue(kpis[0], 0, avg, 1500);
        animateValue(kpis[1], 0, latestPredictions.filter(p => p.risk_level !== 'Low').length, 1200);
        animateValue(kpis[2], 0, top.final_probability * 100, 1500);
        animateValue(kpis[3], 0, latestAlerts.length, 1000);

        document.getElementById('topRiskRegion').textContent = `${top.region} (Critical)`;
    }

    function plotGeospatialRisk() {
        markersLayer.clearLayers();
        latestPredictions.forEach(p => {
            const coords = REGIONS_DATA[p.region];
            if (coords) {
                const color = p.risk_level === 'High' ? '#ef4444' : (p.risk_level === 'Moderate' ? '#f59e0b' : '#10b981');
                const radius = (p.final_probability * 50) + 12;

                const circle = L.circleMarker([coords.lat, coords.lon], {
                    radius: radius,
                    fillColor: color,
                    color: 'rgba(255,255,255,0.2)',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.75
                }).addTo(markersLayer);

                circle.bindPopup(`
                    <div style="text-align:center; font-family:'Space Mono',monospace; font-size:0.75rem;">
                        <b style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.9rem; display:block; margin-bottom:4px;">${p.region}</b>
                        <span style="color:#60a5fa;">${(p.final_probability * 100).toFixed(1)}% Risk Prob.</span>
                    </div>
                `);
                circle.on('click', () => { currentRegion = p.region; globalSelector.value = p.region; syncAllViews(p.region); });
            }
        });
    }

    function updateRiskOverviewUI(regionName) {
        const title = document.getElementById('telemetryRegionName');
        const list = document.getElementById('indicatorList');
        const banner = document.getElementById('riskBanner');
        const bannerText = document.getElementById('bannerText');

        title.textContent = regionName.toUpperCase();

        if (regionName === "All") {
            list.innerHTML = '<div class="telemetry-placeholder">Select a coastal sector for live telemetry injection.</div>';
            map.flyTo([18, 82], 5);
            bannerText.textContent = "Global surveillance active. Monitoring multi-source satellite arrays.";
        } else {
            const r = latestReadings.find(d => d.region === regionName);
            const p = latestPredictions.find(d => d.region === regionName);
            if (r) {
                list.innerHTML = `
                    <div class="telemetry-row"><span>SST (Satellite)</span><strong>${r.sst} °C</strong></div>
                    <div class="telemetry-row"><span>Wind Intensity</span><strong>${r.wind_speed} kn</strong></div>
                    <div class="telemetry-row"><span>Atm. Pressure</span><strong>${r.pressure} hPa</strong></div>
                    <div class="telemetry-row"><span>Rain Accumulation</span><strong>${r.rainfall} mm</strong></div>
                `;
            }
            if (p) {
                banner.className = `global-advisory-banner ${p.risk_level.toLowerCase()}`;
                bannerText.innerHTML = `ADVISORY: <strong>${p.region.toUpperCase()}</strong> sector currently under <strong>${p.risk_level.toUpperCase()}</strong> risk surveillance.`;
            }
            map.flyTo([REGIONS_DATA[regionName].lat, REGIONS_DATA[regionName].lon], 7);
        }
    }

    // --- Deep Analytics Engine ---
    async function loadAnalyticsData() {
        const region = currentRegion === "All" ? latestPredictions[0].region : currentRegion;
        document.getElementById('analyticsRegionName').textContent = region;

        try {
            const history = await fetch(`${API_BASE}/history/${encodeURIComponent(region)}`).then(res => res.json());
            if (history.length > 0) {
                renderIntelligenceCharts(history, region);
                generateInsights(history, region);
            }
        } catch (e) { console.error('Analytics Fetch Error:', e); }
    }

    function generateInsights(data, region) {
        const peak = Math.max(...data.map(d => d.final_probability)) * 100;
        const latestP = data[data.length - 1].raw_pressure;
        const avgP = data.reduce((acc, d) => acc + d.raw_pressure, 0) / data.length;
        const dev = (latestP - avgP).toFixed(1);

        document.getElementById('histPeakRisk').textContent = `${peak.toFixed(1)}%`;
        document.getElementById('pressureTrend').textContent = `${dev > 0 ? '+' : ''}${dev} hPa`;
        document.getElementById('windIntensity').textContent = `${Math.max(...data.map(d => d.raw_wind))} kn`;

        document.getElementById('intelligenceSummary').innerHTML =
            `Analysis for <strong>${region}</strong>: Current pressure deviation of ${dev} hPa against a 30-day peak risk of ${peak.toFixed(1)}%
            suggests a ${peak > 40 ? 'highly unstable' : 'nominal'} atmospheric state. Monitoring continues.`;
    }

    // --- Broadcast Engine ---
    async function loadBroadcasts() {
        document.getElementById('broadcastRegionName').textContent = currentRegion;
        latestAlerts = await fetch(`${API_BASE}/alerts`).then(res => res.json());
        renderAlertGrid();
    }

    function renderAlertGrid() {
        const grid = document.getElementById('alertContainer');
        let filtered = latestAlerts;
        if (currentRegion !== "All") filtered = filtered.filter(a => a.region === currentRegion);
        if (currentSeverityFilter !== "all") filtered = filtered.filter(a => a.severity === (currentSeverityFilter === 'high' ? 'High' : 'Moderate'));

        if (filtered.length === 0) {
            grid.innerHTML = `<div class="broadcast-empty">No ${currentSeverityFilter} alerts active for this sector.</div>`;
            return;
        }

        grid.innerHTML = filtered.map(a => `
            <div class="broadcast-card ${a.severity.toLowerCase()} animate-reveal">
                <span class="severity-tag">${a.severity}</span>
                <h4>${a.alert_type}</h4>
                <p>${a.message}</p>
                <div class="card-footer">
                    <span>📍 ${a.region}</span>
                    <span>🕒 ${new Date(a.timestamp).toLocaleTimeString()}</span>
                </div>
            </div>
        `).join('');
    }

    // Pill Filtering
    document.querySelectorAll('.pill').forEach(pill => {
        pill.addEventListener('click', () => {
            document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            currentSeverityFilter = pill.getAttribute('data-severity');
            renderAlertGrid();
        });
    });

    // --- Charts (Dark Theme Config) ---
    let probChart, corrChart;
    function renderIntelligenceCharts(data, region) {
        if (probChart) probChart.destroy();
        if (corrChart) corrChart.destroy();

        const labels = data.map(d => new Date(d.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }));

        const getGradient = (ctx, colorStop) => {
            const gradient = ctx.createLinearGradient(0, 0, 0, 280);
            gradient.addColorStop(0, colorStop);
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
            return gradient;
        };

        const chartDefaults = {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: {
                    display: true, position: 'top', align: 'end',
                    labels: {
                        usePointStyle: true, pointStyle: 'circle', padding: 20,
                        font: { size: 11, weight: '600', family: "'Plus Jakarta Sans', sans-serif" },
                        color: '#4a6fa5'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(6, 13, 31, 0.95)',
                    titleFont: { size: 12, weight: '700', family: "'Plus Jakarta Sans'" },
                    bodyFont: { size: 11, family: "'Plus Jakarta Sans'" },
                    padding: 12, cornerRadius: 8,
                    borderColor: 'rgba(37,99,235,0.3)', borderWidth: 1,
                    displayColors: true
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#2d4a72', font: { size: 10 } },
                    border: { color: 'rgba(255,255,255,0.04)' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
                    ticks: { color: '#2d4a72', font: { size: 10 }, padding: 10 },
                    border: { color: 'transparent' }
                }
            }
        };

        const ctx1 = document.getElementById('probabilityChart').getContext('2d');
        probChart = new Chart(ctx1, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Risk Probability (%)',
                    data: data.map(d => (d.final_probability * 100).toFixed(1)),
                    borderColor: '#3b82f6',
                    borderWidth: 2.5,
                    backgroundColor: getGradient(ctx1, 'rgba(59, 130, 246, 0.25)'),
                    fill: true, tension: 0.45,
                    pointRadius: 0, pointHoverRadius: 5,
                    pointHoverBackgroundColor: '#3b82f6',
                    pointHoverBorderColor: '#060d1f',
                    pointHoverBorderWidth: 2
                }]
            },
            options: chartDefaults
        });

        const ctx2 = document.getElementById('correlationChart').getContext('2d');
        corrChart = new Chart(ctx2, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label: 'Atm. Pressure (hPa)',
                        data: data.map(d => d.raw_pressure),
                        borderColor: '#60a5fa', borderWidth: 2,
                        tension: 0.4, pointRadius: 0, pointHoverRadius: 4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Wind Speed (kn)',
                        data: data.map(d => d.raw_wind),
                        borderColor: '#06b6d4', borderWidth: 2,
                        tension: 0.4, pointRadius: 0, pointHoverRadius: 4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                ...chartDefaults,
                scales: {
                    x: chartDefaults.scales.x,
                    y: {
                        ...chartDefaults.scales.y, position: 'left',
                        title: { display: true, text: 'Pressure', font: { size: 10, weight: '700' }, color: '#4a6fa5' }
                    },
                    y1: {
                        type: 'linear', position: 'right',
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#2d4a72', font: { size: 10 } },
                        border: { color: 'transparent' },
                        title: { display: true, text: 'Wind', font: { size: 10, weight: '700' }, color: '#4a6fa5' }
                    }
                }
            }
        });
    }

    // --- Global Sync Button ---
    document.querySelector('.btn-refresh').addEventListener('click', async (e) => {
        const btn = e.target;
        btn.textContent = 'Syncing...'; btn.disabled = true;
        try { await fetch(`${API_BASE}/sync`, { method: 'POST' }); initSystem(); }
        catch (e) { alert('API Offline.'); }
        finally { btn.textContent = 'Sync Data'; btn.disabled = false; }
    });

    initSystem();
});