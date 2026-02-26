class DemoWidget {
    constructor() {
        this.btnScore = document.getElementById('btn-score-demo');
        this.btnRandom = document.getElementById('btn-random-demo');

        if (!this.btnScore || !this.btnRandom) return;

        this.initEvents();
    }

    initEvents() {
        this.btnScore.addEventListener('click', () => this.scoreTransaction());
        this.btnRandom.addEventListener('click', () => this.loadRandom());
    }

    loadRandom() {
        const types = ['P2P_TRANSFER', 'QR_PAYMENT', 'E_COMMERCE', 'BILL_PAYMENT'];
        const banks = ['VCB', 'TCB', 'BIDV', 'MBB', 'ACB', 'VPB', 'MOMO', 'ZALOPAY', 'VNPAY'];

        const typeEl = document.getElementById('demo-type');
        const amountEl = document.getElementById('demo-amount');
        const bankEl = document.getElementById('demo-bank');
        const bioEl = document.getElementById('demo-bio');
        const timeEl = document.getElementById('demo-time');

        // 50% threat, 50% legitimate
        const r = Math.random();

        if (r < 0.25) {
            // Bio Evasion Scenario
            typeEl.value = 'P2P_TRANSFER';
            amountEl.value = Math.floor(9500000 + Math.random() * 5000000);
            bioEl.value = 'false';
            timeEl.value = '03:15';
        } else if (r < 0.5) {
            // Fake Shipper Scenario
            typeEl.value = 'P2P_TRANSFER';
            amountEl.value = Math.floor(30000 + Math.random() * 15000);
            bioEl.value = 'true';
            timeEl.value = '14:30';
        } else {
            // Legitimate
            typeEl.value = Math.random() > 0.5 ? 'BILL_PAYMENT' : 'QR_PAYMENT';
            amountEl.value = Math.floor(100000 + Math.random() * 800000);
            bioEl.value = 'true';
            timeEl.value = '10:45';
        }

        bankEl.value = banks[Math.floor(Math.random() * banks.length)];

        // Hide result panel to prompt fresh score
        const resultPanel = document.getElementById('demo-result-panel');
        if (resultPanel) resultPanel.style.display = 'none';
    }

    scoreTransaction() {
        const type = document.getElementById('demo-type').value;
        const amount = parseFloat(document.getElementById('demo-amount').value);
        const bio = document.getElementById('demo-bio').value === 'true';
        const time = document.getElementById('demo-time').value;

        // Deterministic mock scoring engine (matching real XGBoost behavior)
        let score = 0.05; // Base low
        let features = [];

        // Setup base features that push score down (teal)
        features.push({ name: 'Lịch sử giao dịch cặp (Snd-Rcv)', val: -0.15 });
        features.push({ name: 'Độ tuổi thiết bị (h)', val: -0.12 });
        features.push({ name: 'Lịch sử VCB 30 ngày', val: -0.05 });

        // Threat Logic
        // 1. Bio evasion (large amount, no bio, weird hour)
        if (!bio && amount > 9000000) {
            score = 0.88 + (Math.random() * 0.1);
            features = [
                { name: 'Xác thực sinh trắc học thiếu', val: +0.65 },
                { name: 'Số tiền > Khung giờ trung bình', val: +0.42 },
                { name: 'Vận tốc thiết bị (24h)', val: +0.28 },
                { name: 'Lịch sử giao dịch cặp (Snd-Rcv)', val: -0.10 },
                { name: 'Độ tuổi thiết bị (h)', val: -0.05 }
            ];
        }
        // 2. Fake Shipper (P2P + tiny amount)
        else if (type === 'P2P_TRANSFER' && amount < 55000) {
            score = 0.82 + (Math.random() * 0.08);
            features = [
                { name: 'Tần suất nhận tiền (1h) cao', val: +0.55 },
                { name: 'Số tiền siêu nhỏ (Micro-tx)', val: +0.38 },
                { name: 'Mạng lưới chuyển tiền đa đồ thị', val: +0.25 },
                { name: 'Xác thực sinh trắc học', val: -0.10 },
                { name: 'Địa chỉ IP tĩnh nội địa', val: -0.05 }
            ];
        }
        // 3. Normal / Standard
        else {
            score = 0.02 + (Math.random() * 0.07);
            features = [
                { name: 'Lịch sử giao dịch cặp (Snd-Rcv)', val: -0.45 },
                { name: 'Thiết bị & IP quen thuộc', val: -0.32 },
                { name: 'Khung giờ hành chính', val: -0.22 },
                { name: 'Tốc độ giao dịch (1h) bình thường', val: -0.15 },
                { name: 'Loại Giao Dịch P2P', val: +0.02 }
            ];
        }

        this.renderResult(score, features);
    }

    renderResult(score, features) {
        const resultPanel = document.getElementById('demo-result-panel');
        resultPanel.style.display = 'block';

        // Re-trigger CSS animation
        resultPanel.style.animation = 'none';
        resultPanel.offsetHeight; /* trigger reflow */
        resultPanel.style.animation = null;

        // Top Metrics
        const scoreDisplay = document.getElementById('demo-score-val');
        const decisionBadge = document.getElementById('demo-decision-badge');

        scoreDisplay.innerText = score.toFixed(4);

        // Colors
        if (score >= 0.8) {
            scoreDisplay.style.color = 'var(--threat-red)';
            decisionBadge.className = 'decision-badge bg-flagged';
            decisionBadge.innerText = 'FLAGGED FOR REVIEW';
        } else {
            scoreDisplay.style.color = 'var(--safe-teal)';
            decisionBadge.className = 'decision-badge bg-legit';
            decisionBadge.innerText = 'LEGITIMATE';
        }

        // Diagnostics
        document.getElementById('diag-tx-count').innerText = (score > 0.8) ? Math.floor(Math.random() * 15 + 5) : Math.floor(Math.random() * 2 + 1);
        document.getElementById('diag-rcv-count').innerText = (score > 0.8 && features[0].val === 0.55) ? Math.floor(Math.random() * 40 + 20) : Math.floor(Math.random() * 3 + 1);
        document.getElementById('diag-pair').innerText = (score > 0.8) ? '0' : Math.floor(Math.random() * 15 + 5);
        document.getElementById('diag-bio-val').innerText = document.getElementById('demo-bio').value === 'true' ? 'Yes' : 'No';

        // Latency
        document.getElementById('demo-latency').innerText = `Scored in ${Math.floor(25 + Math.random() * 30)}ms`;

        // Render SHAP
        const shapList = document.getElementById('demo-shap-list');
        shapList.innerHTML = ''; // clear

        // Find max absolute value to scale bars properly
        const maxAbs = Math.max(...features.map(f => Math.abs(f.val)));

        features.forEach(f => {
            const row = document.createElement('div');
            row.className = 'shap-bar-row';

            const pct = (Math.abs(f.val) / maxAbs) * 100;
            const color = f.val > 0 ? 'var(--threat-red)' : 'var(--safe-teal)';
            const align = f.val > 0 ? 'left: 50%;' : `right: 50%; width: ${pct / 2}%;`;
            const fillWidth = f.val > 0 ? `width: ${pct / 2}%;` : '';

            row.innerHTML = `
                <div class="shap-label" title="${f.name}">${f.name}</div>
                <div class="shap-track">
                    <div class="shap-fill" style="background:${color}; ${align} ${fillWidth}"></div>
                </div>
                <div class="shap-val">${f.val > 0 ? '+' : ''}${f.val.toFixed(3)}</div>
            `;
            shapList.appendChild(row);
        });
    }
}

window.DemoWidget = DemoWidget;
