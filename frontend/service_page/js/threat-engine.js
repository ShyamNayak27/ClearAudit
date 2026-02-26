class ThreatEngineRenderer {
    constructor(canvasId, ambientId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d', { alpha: true });

        this.ambientCanvas = document.getElementById(ambientId);
        this.ambientCtx = this.ambientCanvas ? this.ambientCanvas.getContext('2d', { alpha: false }) : null;

        this.width = 0;
        this.height = 0;
        this.currentFrame = 0;
        this.totalFrames = 300;

        // Node Definitions (Vietnam Geography Approx)
        this.nodes = [
            { id: 'vcb', name: 'VCB', x: 0.45, y: 0.25, type: 'hanoi' },     // Hanoi
            { id: 'tcb', name: 'TCB', x: 0.48, y: 0.28, type: 'hanoi' },
            { id: 'bidv', name: 'BIDV', x: 0.42, y: 0.27, type: 'hanoi' },
            { id: 'mbb', name: 'MBB', x: 0.46, y: 0.30, type: 'hanoi' },
            { id: 'vpb', name: 'VPB', x: 0.60, y: 0.55, type: 'danang' },    // Da Nang
            { id: 'momo', name: 'MOMO', x: 0.55, y: 0.85, type: 'hcmc' },    // HCMC
            { id: 'acb', name: 'ACB', x: 0.58, y: 0.82, type: 'hcmc' },
            { id: 'zalo', name: 'ZALOPAY', x: 0.52, y: 0.88, type: 'hcmc' },
            { id: 'vnpay', name: 'VNPAY', x: 0.50, y: 0.84, type: 'hcmc' },
        ];

        // Edges
        this.edges = [
            { from: 0, to: 4 }, { from: 1, to: 4 }, { from: 2, to: 0 },
            { from: 3, to: 1 }, { from: 4, to: 5 }, { from: 4, to: 6 },
            { from: 5, to: 7 }, { from: 6, to: 8 }, { from: 8, to: 1 },
            { from: 7, to: 2 }
        ];

        // Particles
        this.particles = [];
        for (let i = 0; i < 40; i++) {
            this.particles.push({
                edgeIdx: Math.floor(Math.random() * this.edges.length),
                progress: Math.random(),
                speed: 0.005 + Math.random() * 0.01,
            });
        }

        this.resize();
        window.addEventListener('resize', this.debounce(() => this.resize(), 150));
    }

    debounce(func, wait) {
        let timeout;
        return function (...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func(...args), wait);
        };
    }

    resize() {
        const parent = this.canvas.parentElement;
        this.width = parent.clientWidth;
        this.height = parent.clientHeight;

        // High DPI Support
        const dpr = window.devicePixelRatio || 1;

        this.canvas.width = this.width * dpr;
        this.canvas.height = this.height * dpr;
        this.ctx.scale(dpr, dpr);

        if (this.ambientCanvas) {
            this.ambientCanvas.width = this.width * dpr;
            this.ambientCanvas.height = this.height * dpr;
            this.ambientCtx.scale(dpr, dpr);
        }

        // Colors
        this.colors = {
            bg: '#07080d',
            nodeSafe: '#00c4cc',
            nodeThreat: '#e5141f',
            edge: 'rgba(143, 168, 200, 0.15)',
            particleSafe: '#00c4cc',
            particleThreat: '#e5141f',
            text: '#8fa8c8'
        };

        // Precalculate pixel pos
        this.pNodes = this.nodes.map(n => ({
            ...n,
            px: n.x * this.width,
            py: n.y * this.height
        }));

        this.drawFrame(this.currentFrame);
    }

    setFrame(frame) {
        this.currentFrame = Math.max(0, Math.min(this.totalFrames, frame));
        this.drawFrame(this.currentFrame);
    }

    play() {
        let lastTime = 0;
        const fps = 30; // Control speed of animation

        const loop = (timestamp) => {
            if (!lastTime) lastTime = timestamp;
            const progress = timestamp - lastTime;

            if (progress > (1000 / fps)) {
                this.currentFrame += 1;
                if (this.currentFrame > this.totalFrames) {
                    this.currentFrame = 0; // Loop continuous playback
                }
                this.drawFrame(this.currentFrame);
                lastTime = timestamp;
            }
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }

    drawFrame(f) {
        this.ctx.clearRect(0, 0, this.width, this.height);

        // Draw Map Silhouette (simplified abstract polygon)
        this.drawVietnamSilhouette(f);

        // Update Particles
        this.particles.forEach(p => {
            p.progress += p.speed;
            if (p.progress > 1) p.progress = 0;
        });

        // Draw Edges
        this.drawEdges(f);

        // Draw Nodes
        this.drawNodes(f);

        // Draw Fraud Waves (120 - 200)
        if (f > 110 && f < 230) {
            this.drawFraudWaves(f);
        }

        // Draw Detection Ring (200 - 260)
        if (f > 200) {
            this.drawDetectionSweep(f);
        }

        // Sync ambient canvas
        if (this.ambientCanvas) {
            // Draw background explicitly for ambient so color bleeds
            this.ambientCtx.fillStyle = this.colors.bg;
            this.ambientCtx.fillRect(0, 0, this.width, this.height);
            this.ambientCtx.drawImage(this.canvas, 0, 0, this.width, this.height);
        }
    }

    drawVietnamSilhouette(f) {
        const alpha = Math.min(1, f / 30); // Fade in map 0-30
        if (alpha <= 0) return;

        this.ctx.save();
        this.ctx.globalAlpha = alpha * 0.4;
        this.ctx.fillStyle = '#0a0b12';
        this.ctx.beginPath();
        this.ctx.moveTo(this.width * 0.4, this.height * 0.1);
        this.ctx.lineTo(this.width * 0.6, this.height * 0.15);
        this.ctx.lineTo(this.width * 0.55, this.height * 0.4);
        this.ctx.lineTo(this.width * 0.65, this.height * 0.6);
        this.ctx.lineTo(this.width * 0.6, this.height * 0.9);
        this.ctx.lineTo(this.width * 0.45, this.height * 0.95);
        this.ctx.lineTo(this.width * 0.48, this.height * 0.7);
        this.ctx.lineTo(this.width * 0.38, this.height * 0.4);
        this.ctx.closePath();
        this.ctx.fill();
        this.ctx.lineWidth = 1;
        this.ctx.strokeStyle = 'rgba(143, 168, 200, 0.05)';
        this.ctx.stroke();
        this.ctx.restore();
    }

    drawEdges(f) {
        const alpha = Math.min(1, Math.max(0, (f - 30) / 30)); // Fade in 30-60
        if (alpha <= 0) return;

        this.ctx.save();
        this.ctx.lineWidth = 1;
        this.edges.forEach(e => {
            const n1 = this.pNodes[e.from];
            const n2 = this.pNodes[e.to];

            this.ctx.strokeStyle = this.colors.edge;
            this.ctx.globalAlpha = alpha;
            this.ctx.beginPath();
            this.ctx.moveTo(n1.px, n1.py);
            // Curve
            const cx = (n1.px + n2.px) / 2 + 50;
            const cy = (n1.py + n2.py) / 2;
            this.ctx.quadraticCurveTo(cx, cy, n2.px, n2.py);
            this.ctx.stroke();
        });
        this.ctx.restore();

        // Draw Particles
        if (f > 60) {
            this.ctx.save();
            this.particles.forEach((p, idx) => {
                const e = this.edges[p.edgeIdx];
                const n1 = this.pNodes[e.from];
                const n2 = this.pNodes[e.to];

                const cx = (n1.px + n2.px) / 2 + 50;
                const cy = (n1.py + n2.py) / 2;

                const t = p.progress;
                // quadratic bezier
                const x = Math.pow(1 - t, 2) * n1.px + 2 * (1 - t) * t * cx + Math.pow(t, 2) * n2.px;
                const y = Math.pow(1 - t, 2) * n1.py + 2 * (1 - t) * t * cy + Math.pow(t, 2) * n2.py;

                // Threat overrides
                let isThreat = false;
                if (f > 120 && f < 220) {
                    if (f > 130 && p.edgeIdx % 3 === 0) isThreat = true;
                    if (f > 150 && p.edgeIdx % 2 === 0) isThreat = true;
                    if (f > 170 && p.edgeIdx % 4 === 0) isThreat = true;
                }

                const pAlpha = (f > 260) ? 0.3 : 1; // dim after secure

                this.ctx.globalAlpha = pAlpha;
                this.ctx.fillStyle = isThreat ? this.colors.particleThreat : this.colors.particleSafe;
                this.ctx.shadowBlur = 10;
                this.ctx.shadowColor = this.ctx.fillStyle;

                this.ctx.beginPath();
                this.ctx.arc(x, y, 2, 0, Math.PI * 2);
                this.ctx.fill();
                this.ctx.shadowBlur = 0; // reset
            });
            this.ctx.restore();
        }
    }

    drawNodes(f) {
        const alpha = Math.min(1, f / 60);
        if (alpha <= 0) return;

        this.ctx.save();
        this.ctx.globalAlpha = alpha;

        this.pNodes.forEach((n, i) => {
            // Determine state
            let color = this.colors.nodeSafe;
            let r = 4;

            // Threat state
            if (f > 120 && f < 230) {
                if (f > 120 && i === 0) { color = this.colors.nodeThreat; r = 6 + Math.sin(f) * 2; }
                if (f > 140 && i === 5) { color = this.colors.nodeThreat; r = 6 + Math.sin(f) * 2; }
                if (f > 160 && (i === 4 || i === 7)) { color = this.colors.nodeThreat; r = 5 + Math.sin(f) * 2; }
                if (f > 180 && i % 2 !== 0) { color = this.colors.nodeThreat; r = 5 + Math.sin(f); }
            }

            this.ctx.fillStyle = color;
            this.ctx.shadowBlur = (color === this.colors.nodeThreat) ? 20 : 10;
            this.ctx.shadowColor = color;

            this.ctx.beginPath();
            this.ctx.arc(n.px, n.py, r, 0, Math.PI * 2);
            this.ctx.fill();

            // Labels (only visible safely or brightly if threat)
            this.ctx.shadowBlur = 0;
            this.ctx.font = "10px 'DM Mono'";
            this.ctx.fillStyle = (f > 260) ? color : this.colors.text;
            this.ctx.fillText(n.name, n.px + 10, n.py + 4);
        });
        this.ctx.restore();
    }

    drawFraudWaves(f) {
        this.ctx.save();

        // Wave 1: Fake Shipper (Node 0) at f=125
        if (f > 125 && f < 170) this.drawGlyph(this.pNodes[0].px - 40, this.pNodes[0].py - 20, 'shipper', Math.min(1, (f - 125) / 10));
        // Wave 2: Quishing (Node 5) at f=150
        if (f > 145 && f < 190) this.drawGlyph(this.pNodes[5].px + 20, this.pNodes[5].py + 20, 'qr', Math.min(1, (f - 145) / 10));
        // Wave 3: Biometric (Node 4) at f=170
        if (f > 165 && f < 210) this.drawGlyph(this.pNodes[4].px - 30, this.pNodes[4].py, 'bio', Math.min(1, (f - 165) / 10));
        // Wave 4: Tet (Node 7) at f=190
        if (f > 185 && f < 230) this.drawGlyph(this.pNodes[7].px - 20, this.pNodes[7].py - 40, 'tet', Math.min(1, (f - 185) / 10));

        this.ctx.restore();
    }

    drawGlyph(x, y, type, alpha) {
        this.ctx.strokeStyle = this.colors.nodeThreat;
        this.ctx.lineWidth = 2;
        this.ctx.globalAlpha = alpha * 0.8;
        this.ctx.shadowBlur = 10;
        this.ctx.shadowColor = this.colors.nodeThreat;

        const s = 15; // scale

        this.ctx.beginPath();
        if (type === 'shipper') {
            // Pin icon
            this.ctx.arc(x, y - s / 2, s / 2, 0, Math.PI * 2);
            this.ctx.moveTo(x - s / 2, y - s / 2);
            this.ctx.lineTo(x, y + s);
            this.ctx.lineTo(x + s / 2, y - s / 2);
        } else if (type === 'qr') {
            // Grid
            this.ctx.rect(x - s, y - s, s * 2, s * 2);
            this.ctx.moveTo(x, y - s); this.ctx.lineTo(x, y + s);
            this.ctx.moveTo(x - s, y); this.ctx.lineTo(x + s, y);
        } else if (type === 'bio') {
            // Fingerprint arc
            this.ctx.arc(x, y, s, Math.PI, 0);
            this.ctx.arc(x, y, s - 4, Math.PI, 0);
            // Slash
            this.ctx.moveTo(x - s, y - s); this.ctx.lineTo(x + s, y + s);
        } else if (type === 'tet') {
            // Lantern
            this.ctx.ellipse(x, y, s / 1.5, s, 0, 0, Math.PI * 2);
            this.ctx.moveTo(x, y - s - 5); this.ctx.lineTo(x, y + s + 5);
        }
        this.ctx.stroke();
    }

    drawDetectionSweep(f) {
        // 200 - 260: ring expands from center
        const prog = Math.min(1, Math.max(0, (f - 200) / 60));
        if (prog <= 0) return;

        const cx = this.width / 2;
        const cy = this.height / 2;
        const maxRadius = Math.max(this.width, this.height);
        const rad = prog * maxRadius;

        this.ctx.save();

        // Ring line
        this.ctx.beginPath();
        this.ctx.arc(cx, cy, rad, 0, Math.PI * 2);
        this.ctx.lineWidth = 4;
        this.ctx.strokeStyle = this.colors.nodeSafe;
        this.ctx.globalAlpha = (1 - prog) * 0.8;
        this.ctx.shadowBlur = 20;
        this.ctx.shadowColor = this.colors.nodeSafe;
        this.ctx.stroke();

        // Lock text at end
        if (f > 250) {
            const lockProg = Math.min(1, (f - 250) / 20);
            this.ctx.fillStyle = this.colors.nodeSafe;
            this.ctx.globalAlpha = lockProg;
            this.ctx.font = "bold 24px 'DM Mono'";
            this.ctx.textAlign = "center";
            this.ctx.shadowBlur = 10;
            this.ctx.fillText("SYSTEM SECURE", cx, cy - 100);
        }

        this.ctx.restore();
    }
}

// Export for module usage (or global attach)
window.ThreatEngineRenderer = ThreatEngineRenderer;
