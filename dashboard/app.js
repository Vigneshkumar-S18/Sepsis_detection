// Clinician Dashboard Javascript Logic Manager

document.addEventListener("DOMContentLoaded", () => {
    // 1. Session Login handler
    const loginScreen = document.getElementById("login-screen");
    const loginForm = document.getElementById("login-form");
    const mainInterface = document.getElementById("main-interface");
    const profileName = document.getElementById("profile-name");
    const profileRole = document.getElementById("profile-role");
    
    let activePatientId = null;
    let timelineChart = null;

    loginForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const username = document.getElementById("username").value;
        const role = document.getElementById("user-role").value;
        
        // Update user profiles
        profileName.textContent = username;
        profileRole.textContent = role;
        
        // Hide login and reveal main dashboard
        loginScreen.classList.add("opacity-0", "pointer-events-none");
        setTimeout(() => loginScreen.classList.add("hidden"), 500);
        mainInterface.classList.remove("hidden");
        
        // Start live polling intervals
        startPolling();
    });

    // 2. Poll API endpoints every 3 seconds
    function startPolling() {
        fetchDashboardStats();
        fetchPatientsList();
        
        setInterval(() => {
            fetchDashboardStats();
            fetchPatientsList();
            if (activePatientId) {
                // Refresh active patient details
                refreshActivePatientDetails();
            }
        }, 3000);
    }

    // 3. Fetch Dashboard Stats
    async function fetchDashboardStats() {
        try {
            const res = await fetch("/api/dashboard");
            if (res.status === 200) {
                const data = await res.json();
                document.getElementById("stat-occupancy").textContent = data.occupancy;
                document.getElementById("stat-high-risk").textContent = data.high_risk_count;
                document.getElementById("stat-alerts").textContent = data.active_alerts;
                document.getElementById("stat-avg-risk").textContent = `${(data.average_risk * 100).toFixed(1)}%`;
            }
        } catch (e) {
            console.error("Error fetching stats:", e);
        }
    }

    // 4. Fetch Patients Ward Grid
    async function fetchPatientsList() {
        try {
            const res = await fetch("/api/patients");
            if (res.status === 200) {
                const patients = await res.json();
                renderPatientGrid(patients);
            }
        } catch (e) {
            console.error("Error fetching patients list:", e);
        }
    }

    // 5. Render Patient Cards
    function renderPatientGrid(patients) {
        const grid = document.getElementById("patient-grid");
        if (patients.length === 0) {
            grid.innerHTML = `<div class="text-slate-500 text-sm text-center py-10">No telemetry data ingested yet...</div>`;
            return;
        }

        grid.innerHTML = "";
        patients.forEach((pat) => {
            const riskPercent = (pat.latest_prediction.risk * 100).toFixed(0);
            const isHigh = pat.latest_prediction.risk >= 0.70;
            const isMed = pat.latest_prediction.risk >= 0.50 && pat.latest_prediction.risk < 0.70;
            
            let statusColor = "text-emerald-400";
            let borderColor = "border-teal-950/40";
            let glowClass = "";
            let riskLabel = "LOW";
            
            if (isHigh) {
                statusColor = "text-rose-400 animate-pulse";
                borderColor = "border-rose-500/40";
                glowClass = "glow-risk-high";
                riskLabel = "CRITICAL";
            } else if (isMed) {
                statusColor = "text-amber-400";
                borderColor = "border-amber-500/40";
                glowClass = "glow-risk-medium";
                riskLabel = "MODERATE";
            }

            const activeClass = (activePatientId === pat.patient_id) ? "bg-slate-900/90 ring-1 ring-teal-500/50" : "bg-slate-900/40 hover:bg-slate-900/60";

            const card = document.createElement("div");
            card.className = `p-4 rounded-xl border ${borderColor} ${activeClass} ${glowClass} cursor-pointer transition-all duration-200`;
            card.innerHTML = `
                <div class="flex justify-between items-start">
                    <div>
                        <h4 class="font-bold text-slate-100">${pat.name}</h4>
                        <span class="text-xs text-slate-400">Bed ${pat.patient_id.replace(/\D/g, '') || pat.patient_id}</span>
                    </div>
                    <div class="text-right">
                        <span class="text-xs font-semibold ${statusColor}">${riskLabel}</span>
                        <div class="text-2xl font-bold text-slate-200 mt-0.5">${riskPercent}%</div>
                    </div>
                </div>
                <div class="mt-4 flex items-center justify-between text-xs text-slate-400">
                    <div>Stay: <span class="text-slate-300 font-medium">${pat.latest_vitals.ICULOS}h</span></div>
                    <div class="flex items-center space-x-1.5">
                        <span class="w-2.5 h-2.5 rounded-full ${pat.alert_triggered ? 'bg-amber-400 animate-ping' : 'bg-teal-500'}"></span>
                        <span>${pat.alert_triggered ? 'ACTIVE ALERT' : 'STABLE'}</span>
                    </div>
                </div>
            `;
            
            card.addEventListener("click", () => {
                activePatientId = pat.patient_id;
                // Highlight and load details
                document.querySelectorAll("#patient-grid > div").forEach(c => c.classList.remove("ring-1", "ring-teal-500/50", "bg-slate-900/90"));
                card.classList.add("ring-1", "ring-teal-500/50", "bg-slate-900/90");
                loadPatientDetails(pat);
            });
            
            grid.appendChild(card);
        });
    }

    // 6. Load Patient Details View
    function loadPatientDetails(pat) {
        document.getElementById("details-placeholder").classList.add("hidden");
        const content = document.getElementById("details-content");
        content.classList.remove("hidden");

        // Set static text metrics
        document.getElementById("patient-title").textContent = `Patient ${pat.patient_id.toUpperCase()}`;
        document.getElementById("pat-age").textContent = pat.age.toFixed(1);
        document.getElementById("pat-gender").textContent = pat.gender;
        document.getElementById("pat-hours").textContent = pat.latest_vitals.ICULOS;

        // Vitals
        document.getElementById("val-hr").textContent = `${pat.latest_vitals.HR.toFixed(1)} bpm`;
        document.getElementById("val-map").textContent = `${pat.latest_vitals.MAP.toFixed(1)} mmHg`;
        document.getElementById("val-temp").textContent = `${pat.latest_vitals.Temp.toFixed(1)} °C`;
        document.getElementById("val-resp").textContent = `${pat.latest_vitals.Resp.toFixed(1)} /min`;
        document.getElementById("val-spo2").textContent = `${pat.latest_vitals.SpO2.toFixed(1)}%`;

        // Style status badge
        const badge = document.getElementById("patient-badge");
        badge.textContent = pat.latest_prediction.status.toUpperCase();
        if (pat.latest_prediction.risk >= 0.70) {
            badge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold tracking-wider bg-rose-500/20 text-rose-300 border border-rose-500/30";
        } else if (pat.latest_prediction.risk >= 0.50) {
            badge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/30";
        } else {
            badge.className = "px-2.5 py-0.5 rounded-full text-xs font-semibold tracking-wider bg-teal-500/20 text-teal-300 border border-teal-500/30";
        }

        // Configure acknowledgement button
        const btnAck = document.getElementById("btn-ack");
        if (pat.alert_triggered) {
            btnAck.style.display = "block";
            btnAck.onclick = () => acknowledgeAlert(pat.patient_id);
        } else {
            btnAck.style.display = "none";
        }

        // Render line graph and local attributions
        fetchTimelineAndExplainability(pat.patient_id);
    }

    // 7. Fetch timeline history and explanations
    async function fetchTimelineAndExplainability(patient_id) {
        try {
            // A. Fetch Timeline History
            const resHist = await fetch(`/api/prediction/${patient_id}/history`);
            const history = await resHist.json();
            
            // B. Fetch IG explanation
            const resExp = await fetch(`/api/explanation/${patient_id}`);
            const explanation = await resExp.json();

            // C. Render Chart
            renderRiskTimelineChart(history);
            
            // D. Render IG attributions
            renderExplainabilityBars(explanation);
            
            // E. Fetch recommendation from prediction endpoint directly to maintain consistency
            const resPred = await fetch(`/api/prediction/${patient_id}`);
            const pred = await resPred.json();
            document.getElementById("clinical-rec").textContent = pred.recommendation;

        } catch (e) {
            console.error("Error fetching detailed metrics:", e);
        }
    }

    // 8. Render Risk Timeline Chart (Chart.js)
    function renderRiskTimelineChart(history) {
        const ctx = document.getElementById("timelineChart").getContext("2d");
        
        // Prepare datasets
        const labels = history.map((h, i) => `Hour ${i+1}`);
        const data = history.map(h => h.risk * 100);

        if (timelineChart) {
            timelineChart.destroy();
        }

        timelineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Sepsis Risk %',
                    data: data,
                    borderColor: '#14b8a6',
                    backgroundColor: 'rgba(20, 184, 166, 0.08)',
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.3,
                    pointBackgroundColor: '#14b8a6',
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        ticks: { color: '#64748b', font: { family: 'Outfit' } },
                        grid: { color: 'rgba(51, 65, 85, 0.2)' }
                    },
                    x: {
                        ticks: { color: '#64748b', font: { family: 'Outfit' } },
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    // 9. Render Explainability attributions (Online Integrated Gradients)
    function renderExplainabilityBars(explanation) {
        const container = document.getElementById("attribution-bars");
        container.innerHTML = "";
        
        const topFeatures = explanation.top_features;
        const scores = explanation.attribution_scores;
        
        // Normalize scores for rendering percentage heights
        const maxScore = Math.max(...scores) || 1.0;

        topFeatures.forEach((feat, idx) => {
            const score = scores[idx];
            const heightPercent = Math.min((score / maxScore) * 100, 100).toFixed(0);
            
            const bar = document.createElement("div");
            bar.className = "flex flex-col items-center";
            bar.innerHTML = `
                <div class="h-28 w-full bg-slate-950/80 rounded-lg flex items-end overflow-hidden border border-slate-800 relative">
                    <div class="w-full bg-teal-500/35 border-t border-teal-400/60 rounded-b transition-all duration-500 transition-height" style="height: ${heightPercent}%"></div>
                    <div class="absolute inset-0 flex items-center justify-center text-[10px] text-teal-300 font-mono">${score.toFixed(2)}</div>
                </div>
                <span class="text-[10px] text-slate-400 text-center font-medium mt-2 max-w-[80px] truncate" title="${feat}">${feat}</span>
            `;
            container.appendChild(bar);
        });
    }

    // 10. Acknowledge Alert Handler
    async function acknowledgeAlert(patient_id) {
        try {
            const res = await fetch(`/api/alerts/acknowledge/${patient_id}`, { method: "POST" });
            if (res.status === 200) {
                // Refresh lists immediately
                fetchDashboardStats();
                fetchPatientsList();
                // Refresh active patient badge
                refreshActivePatientDetails();
            }
        } catch (e) {
            console.error("Error acknowledging alert:", e);
        }
    }

    // 11. Helper to refresh details for currently active patient
    async function refreshActivePatientDetails() {
        if (!activePatientId) return;
        try {
            const res = await fetch("/api/patients");
            if (res.status === 200) {
                const patients = await res.json();
                const activePat = patients.find(p => p.patient_id === activePatientId);
                if (activePat) {
                    loadPatientDetails(activePat);
                }
            }
        } catch (e) {
            console.error("Error refreshing active details:", e);
        }
    }
});
