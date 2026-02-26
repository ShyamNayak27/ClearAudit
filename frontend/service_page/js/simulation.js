// ClearAudit - Idea 4 Simulation Dashboard Logic

document.addEventListener('DOMContentLoaded', () => {

    // ─── TAB NAVIGATION ──────────────────────────────────────
    const tabs = document.querySelectorAll('.sim-tab');
    const panels = document.querySelectorAll('.sim-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const target = document.getElementById(tab.getAttribute('data-target'));
            if (target) {
                target.classList.add('active');
            }
        });
    });

    // ─── CHART.JS INITIALIZATION ─────────────────────────────

    Chart.defaults.color = '#8fa8c8';
    Chart.defaults.font.family = "'DM Mono', monospace";
    Chart.defaults.plugins.tooltip.backgroundColor = '#0d1b3e';
    Chart.defaults.plugins.tooltip.titleFont = { family: "'Syne', sans-serif", size: 14 };

    // 1. Precision-Recall Curve (Mocked data representing typical XGBoost hold-out)
    const ctxPr = document.getElementById('chart-pr');
    if (ctxPr) {
        new Chart(ctxPr, {
            type: 'line',
            data: {
                labels: ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
                datasets: [{
                    label: 'Ensemble (Champion)',
                    data: [1.0, 0.99, 0.98, 0.97, 0.96, 0.94, 0.92, 0.90, 0.88, 0.70, 0.0],
                    borderColor: '#00c4cc',
                    backgroundColor: 'rgba(0, 196, 204, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'XGBoost Only',
                    data: [1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.85, 0.82, 0.65, 0.0],
                    borderColor: '#8fa8c8',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Recall' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { title: { display: true, text: 'Precision' }, min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    }

    // 2. Typology Breakdown (Reflecting metrics_report.json)
    const ctxPie = document.getElementById('chart-pie');
    if (ctxPie) {
        new Chart(ctxPie, {
            type: 'doughnut',
            data: {
                labels: ['Fake Shipper COD', 'Biometric Evasion', 'Quishing Abuse', 'Account Takeover / Other'],
                datasets: [{
                    data: [8509, 7035, 5152, 7751],
                    backgroundColor: ['#e5141f', '#00c4cc', '#0d1b3e', '#8fa8c8'],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: { position: 'right' }
                }
            }
        });
    }

    // 3. Hourly Incident Pattern
    const ctxHourly = document.getElementById('chart-hourly');
    if (ctxHourly) {
        new Chart(ctxHourly, {
            type: 'bar',
            data: {
                labels: ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
                datasets: [{
                    label: 'Blocked Transactions',
                    data: [124, 452, 189, 450, 680, 890, 520, 240], // Peak around 3am (ATO/Bio) and 3pm (Shipper)
                    backgroundColor: '#e5141f',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    // ─── LIVE SCORING LOGIC ──────────────────────────────────

    const btnScore = document.getElementById('btn-score');
    const btnRandom = document.getElementById('btn-random');

    const inpSender = document.getElementById('inp-sender');
    const inpRecv = document.getElementById('inp-recv');
    const inpAmount = document.getElementById('inp-amount');
    const inpBank = document.getElementById('inp-bank');
    const inpType = document.getElementById('inp-type');
    const inpGeo = document.getElementById('inp-geo');
    const inpBio = document.getElementById('inp-bio');

    const resultCard = document.getElementById('result-card');

    // RANDOMIZER
    if (btnRandom) {
        btnRandom.addEventListener('click', () => {
            const types = ['P2P_TRANSFER', 'QR_PAYMENT', 'E_COMMERCE', 'BILL_PAYMENT'];
            const banks = ['VCB', 'TCB', 'BIDV', 'MBB', 'ACB', 'MOMO', 'ZALO', 'VNPAY'];

            // Randomly pick a threat scenario
            const r = Math.random();
            if (r < 0.3) {
                // Biometric Evasion
                inpType.value = 'P2P_TRANSFER';
                inpAmount.value = Math.floor(9200000 + Math.random() * 700000); // Near 10M
                inpBio.value = 'false';
            } else if (r < 0.6) {
                // Fake Shipper
                inpType.value = 'P2P_TRANSFER';
                inpAmount.value = Math.floor(20000 + Math.random() * 40000); // 20k - 60k
                inpBio.value = 'true';
            } else {
                // High velocity QR
                inpType.value = 'QR_PAYMENT';
                inpAmount.value = Math.floor(100000 + Math.random() * 2000000);
            }

            // Ensure bank option exists
            const randBank = banks[Math.floor(Math.random() * banks.length)];
            let bankExists = false;
            Array.from(inpBank.options).forEach(opt => {
                if (opt.value === randBank) bankExists = true;
            });
            if (!bankExists) {
                const opt = document.createElement('option');
                opt.value = randBank;
                opt.text = randBank;
                inpBank.add(opt);
            }
            inpBank.value = randBank;

            inpSender.value = 'VN_' + Math.floor(Math.random() * 9000 + 1000);
            inpRecv.value = 'VN_' + Math.floor(Math.random() * 9000 + 1000);
            inpGeo.value = ['w3', 'w4', 'w5', 'w6'][Math.floor(Math.random() * 4)];

            resultCard.style.opacity = '0.5';
        });
    }

    // LIVE SCORE FETCH
    if (btnScore) {
        btnScore.addEventListener('click', async () => {

            // UX transition
            btnScore.innerText = 'SCORING...';
            btnScore.disabled = true;
            resultCard.style.opacity = '0.5';

            // Construct payload matching backend TransactionRequest schema
            const payload = {
                transaction_id: "tx_" + Date.now(),
                sender_id: inpSender.value || "anonymous",
                receiver_id: inpRecv.value || "anonymous",
                amount: parseFloat(inpAmount.value) || 0,
                timestamp: new Date().toISOString(),
                bank_routing: inpBank.value,
                transaction_type: inpType.value,
                is_biometric_verified: inpBio.value === 'true',
                device_id: "dev_" + Math.floor(Math.random() * 1000),
                ip_address: "192.168.1." + Math.floor(Math.random() * 255),
                geohash: inpGeo.value
            };

            const startTime = performance.now();

            try {
                // Real fastAPI call
                const res = await fetch('http://127.0.0.1:8000/score', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!res.ok) {
                    throw new Error(`API returned status: ${res.status}`);
                }

                const data = await res.json();
                const latency = Math.floor(performance.now() - startTime);

                // Update UI with real evaluated data
                document.getElementById('res-score').innerText = data.fraud_score.toFixed(4);

                // Color formatting
                if (data.fraud_score >= 0.8) {
                    document.getElementById('res-score').style.color = 'var(--threat-red)';
                    const badge = document.getElementById('res-decision');
                    badge.className = 'badge badge-red';
                    badge.innerText = data.decision || 'FLAGGED';
                } else {
                    document.getElementById('res-score').style.color = 'var(--safe-teal)';
                    const badge = document.getElementById('res-decision');
                    badge.className = 'badge badge-teal';
                    badge.innerText = data.decision || 'LEGITIMATE';
                }

                document.getElementById('res-xgboost').innerText = data.xgboost_score.toFixed(4);
                document.getElementById('res-ae').innerText = data.ae_anomaly_score.toFixed(4);
                document.getElementById('res-latency').innerText = `${latency}ms`;

                // Diagnostics
                if (data.diagnostics) {
                    document.getElementById('diag-tx1h').innerText = data.diagnostics.tx_1h !== undefined ? data.diagnostics.tx_1h : '-';
                    document.getElementById('diag-rcv24h').innerText = data.diagnostics.distinct_receivers_24h !== undefined ? data.diagnostics.distinct_receivers_24h : '-';
                    document.getElementById('diag-pair').innerText = data.diagnostics.pair_history !== undefined ? data.diagnostics.pair_history : '-';
                }
                document.getElementById('diag-bio').innerText = payload.is_biometric_verified ? 'Yes' : 'No';

                // Real SHAP Features
                const shapList = document.getElementById('shap-list');
                shapList.innerHTML = ''; // clear

                if (data.top_features && data.top_features.length > 0) {
                    const maxAbs = Math.max(...data.top_features.map(f => Math.abs(f.contribution)));

                    data.top_features.forEach(f => {
                        const row = document.createElement('div');
                        row.className = 'shap-row';

                        const val = f.contribution;
                        const pct = maxAbs > 0 ? (Math.abs(val) / maxAbs) * 100 : 0;
                        const color = val > 0 ? 'var(--threat-red)' : 'var(--safe-teal)';
                        const align = val > 0 ? 'left: 50%;' : `right: 50%; width: ${pct / 2}%;`;
                        const fillWidth = val > 0 ? `width: ${pct / 2}%;` : '';

                        // Map internal feature names to slightly more readable labels
                        const niceName = f.feature.replace(/_/g, ' ').toUpperCase();

                        row.innerHTML = `
                            <div class="shap-lbl" title="${f.feature}">${niceName}</div>
                            <div class="shap-bar-container">
                                <div class="shap-bar" style="background:${color}; ${align} ${fillWidth}"></div>
                            </div>
                            <div class="shap-amt">${val > 0 ? '+' : ''}${val.toFixed(4)}</div>
                        `;
                        shapList.appendChild(row);
                    });
                } else {
                    shapList.innerHTML = '<div style="color:var(--text-secondary); font-family:var(--font-mono); font-size:0.8rem;">No SHAP explanation available.</div>';
                }

                resultCard.style.opacity = '1';

            } catch (err) {
                console.error("Scoring failed:", err);
                alert("Failed to connect to API. Is `uvicorn src.api.app:app` running on 127.0.0.1:8000?");
            } finally {
                btnScore.innerText = 'EVALUATE TRANSACTION (LIVE API)';
                btnScore.disabled = false;
            }
        });
    }

});
