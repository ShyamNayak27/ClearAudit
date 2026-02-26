/**
 * simulation.js
 * Handles the logic for the ClearAudit Simulation Dashboard.
 */

document.addEventListener('DOMContentLoaded', () => {

    // ============================================================
    // TAB NAVIGATION
    // ============================================================
    const tabs = document.querySelectorAll('.sim-tab');
    const panels = document.querySelectorAll('.sim-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // 1. Remove active class from all
            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            // 2. Add active to clicked tab and corresponding panel
            tab.classList.add('active');
            const targetPanel = document.getElementById(tab.dataset.target);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
        });
    });

    // ============================================================
    // LIVE SCORING LOGIC
    // ============================================================
    const btnScore = document.getElementById('btn-score');
    const btnRandom = document.getElementById('btn-random');
    const resultCard = document.getElementById('scoring-result');
    const shapList = document.getElementById('shap-list');

    // Input fields
    const inpAmount = document.getElementById('inp-amount');
    const inpSender = document.getElementById('inp-sender');
    const inpRecv = document.getElementById('inp-recv');
    const inpBank = document.getElementById('inp-bank');
    const inpType = document.getElementById('inp-type');

    // Result fields
    const resDec = document.getElementById('res-decision');
    const resScore = document.getElementById('res-score');
    const resXgb = document.getElementById('res-xgb');
    const resAe = document.getElementById('res-ae');

    // Diagnostic fields
    const d1h = document.getElementById('d-1h');
    const d24h = document.getElementById('d-24h');
    const drx = document.getElementById('d-rx');
    const ddv = document.getElementById('d-dv');

    btnScore.addEventListener('click', async () => {
        const payload = {
            amount_vnd: parseFloat(inpAmount.value),
            sender_cif: inpSender.value,
            receiver_cif: inpRecv.value,
            receiver_bank_code: inpBank.value,
            transaction_type: inpType.value,
            is_biometric_verified: false,
            device_mac_hash: "web_device",
            timestamp: new Date().toISOString().slice(0, 19).replace('T', ' ')
        };

        btnScore.textContent = 'Scoring via API...';
        btnScore.style.opacity = '0.7';
        resultCard.style.opacity = '0.5';

        try {
            const res = await fetch('../api/score', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) throw new Error('API Error');
            const data = await res.json();

            resScore.textContent = data.fraud_score.toFixed(4);
            resXgb.textContent = data.xgboost_score.toFixed(4);
            resAe.textContent = data.ae_anomaly_score.toFixed(4);

            const dec = data.decision.toUpperCase();
            resDec.textContent = dec;
            if (dec === 'APPROVE') resDec.className = 'badge badge-green';
            else if (dec === 'BLOCK') resDec.className = 'badge badge-red';
            else resDec.className = 'badge badge-orange';

            d1h.textContent = data.diagnostics.tx_1h;
            d24h.textContent = data.diagnostics.tx_24h;
            drx.textContent = data.diagnostics.receivers_24h;
            ddv.textContent = data.diagnostics.devices_24h;

            shapList.innerHTML = data.top_features.map(f => `
                <div class="shap-item">
                    <div class="shap-dot ${f.direction === 'fraud' ? 'fraud' : 'legit'}"></div>
                    <div class="shap-name">${f.feature}</div>
                    <div class="shap-val">${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(4)}</div>
                </div>
            `).join('');

        } catch (e) {
            console.error('API Error:', e);
            resDec.textContent = 'API ERROR';
            resDec.className = 'badge badge-orange';
            resScore.textContent = '-';
            resXgb.textContent = '-';
            resAe.textContent = '-';
            shapList.innerHTML = '<div class="shap-item"><div class="shap-val" style="color:#e5141f;">Failed to reach backend at port 8000.</div></div>';
        }

        resultCard.style.opacity = '1';
        resultCard.style.pointerEvents = 'auto';
        btnScore.textContent = 'Score Transaction';
        btnScore.style.opacity = '1';
    });

    btnRandom.addEventListener('click', () => {
        const banks = ["VCB", "TCB", "BIDV", "VPB", "MBB", "ACB", "MOMO", "ZALOPAY", "VNPAY"];
        const types = ["P2P_TRANSFER", "E_COMMERCE", "QR_PAYMENT", "BILL_PAYMENT"];

        if (Math.random() < 0.3) {
            inpAmount.value = [35000, 50000, 9500000, 12000000][Math.floor(Math.random() * 4)];
            inpType.value = Math.random() > 0.5 ? "P2P_TRANSFER" : types[Math.floor(Math.random() * types.length)];
        } else {
            inpAmount.value = Math.floor(Math.random() * (2000000 - 100000) + 100000);
            inpType.value = types[Math.floor(Math.random() * types.length)];
        }

        const randBank = banks[Math.floor(Math.random() * banks.length)];

        // Ensure options exist in dropdown before selecting
        if (!Array.from(inpBank.options).some(o => o.value === randBank)) {
            const opt = document.createElement('option'); opt.value = randBank; opt.textContent = randBank; inpBank.appendChild(opt);
        }
        inpBank.value = randBank;

        if (!Array.from(inpType.options).some(o => o.value === inpType.value)) {
            const opt = document.createElement('option'); opt.value = inpType.value; opt.textContent = inpType.value; inpType.appendChild(opt);
        }

        inpSender.value = 'VN_' + Math.floor(Math.random() * 9000 + 1000);
        inpRecv.value = 'VN_' + Math.floor(Math.random() * 9000 + 1000);
        resultCard.style.opacity = '0.5';
    });

    // ============================================================
    // CHART.JS RENDERERS
    // ============================================================

    // Set global defaults
    Chart.defaults.font.family = '"DM Mono", monospace';
    Chart.defaults.color = 'rgba(10, 10, 10, 0.6)';
    Chart.defaults.elements.bar.borderRadius = 4;

    // 1. Model Comparison Chart
    const ctxModels = document.getElementById('chart-models');
    if (ctxModels) {
        new Chart(ctxModels, {
            type: 'bar',
            data: {
                labels: ['Ensemble', 'XGBoost', 'Random Forest', 'Autoencoder'],
                datasets: [
                    {
                        label: 'F1-Score',
                        data: [0.9107, 0.9084, 0.8612, 0.1692],
                        backgroundColor: '#00c4cc'
                    },
                    {
                        label: 'Precision',
                        data: [0.9036, 0.8799, 0.8299, 0.4192],
                        backgroundColor: '#0d1b3e'
                    },
                    {
                        label: 'Recall',
                        data: [0.9180, 0.9389, 0.8950, 0.1060],
                        backgroundColor: '#8fa8c8'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 8 } }
                },
                scales: {
                    y: { min: 0.6, max: 1.05 }
                }
            }
        });
    }

    // 2. Fraud Typology Pie Chart
    const ctxPie = document.getElementById('chart-pie');
    if (ctxPie) {
        new Chart(ctxPie, {
            type: 'doughnut',
            data: {
                labels: ['Fake Shipper COD', 'Quishing', 'Biometric Bypass', 'Account Takeover'],
                datasets: [{
                    data: [8509, 5152, 7035, 7751],
                    backgroundColor: ['#00c4cc', '#0d1b3e', '#e5141f', '#8fa8c8'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: { position: 'right', labels: { usePointStyle: true, boxWidth: 8 } }
                }
            }
        });
    }

    // 3. Hourly Distribution Chart
    const ctxHour = document.getElementById('chart-hour');
    if (ctxHour) {
        const hours = Array.from({ length: 24 }, (_, i) => i);
        // Mock legit curve (normal day distribution)
        const legit = hours.map(h => Math.sin((h - 6) * Math.PI / 12) * 50 + 50);
        // Mock fraud curve (spikes at night)
        const fraud = hours.map(h => {
            if (h >= 1 && h <= 5) return Math.random() * 40 + 60; // night spike
            if (h >= 12 && h <= 14) return Math.random() * 20 + 30; // lunch spike
            return Math.random() * 10 + 5;
        });

        new Chart(ctxHour, {
            type: 'line',
            data: {
                labels: hours,
                datasets: [
                    {
                        label: 'Legitimate (%)',
                        data: legit,
                        borderColor: '#00c4cc',
                        backgroundColor: 'rgba(0, 196, 204, 0.1)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Fraud (%)',
                        data: fraud,
                        borderColor: '#e5141f',
                        backgroundColor: 'rgba(229, 20, 31, 0.1)',
                        fill: true,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 8 } }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: { beginAtZero: true, display: false }
                }
            }
        });
    }

    // Trigger initial hit on Score to show the default suspicious block state
    setTimeout(() => btnScore.click(), 500);
});
