/**
 * PipelineEngineRenderer
 * 300-frame procedural Canvas 2D animation showing a data pipeline.
 *
 * Pipeline stages (left to right):
 *   SimPay VN → CTGAN 500k → WGAN-GP 50k → DuckDB Velocity
 *   → XGBoost → Autoencoder → FastAPI → Redis → SHAP Output
 *
 * Frame Phases:
 *   0-80   : Stage nodes appear (staggered), pipes draw in
 *   80-160 : Data particles flow through pipes (teal)
 *   160-240: Fraud anomaly (red particle) enters, propagates, gets flagged at XGBoost
 *   240-299: Detection confirmed, green DETECTED label, system locks
 */
class PipelineEngineRenderer {
    constructor() {
        this.stages = [
            { id: 'simpay', label: 'SimPay VN', shortLabel: 'SIMPAY', col: 0, row: 0 },
            { id: 'ctgan', label: 'CTGAN 500k', shortLabel: 'CTGAN', col: 1, row: -0.3 },
            { id: 'wganGp', label: 'WGAN-GP 50k', shortLabel: 'WGAN-GP', col: 1, row: 0.3 },
            { id: 'duckdb', label: 'DuckDB Velocity', shortLabel: 'DUCKDB', col: 2, row: 0 },
            { id: 'xgboost', label: 'XGBoost', shortLabel: 'XGBOOST', col: 3, row: -0.2 },
            { id: 'autoenc', label: 'Autoencoder', shortLabel: 'AE', col: 3, row: 0.2 },
            { id: 'fastapi', label: 'FastAPI', shortLabel: 'FASTAPI', col: 4, row: 0 },
            { id: 'redis', label: 'Redis', shortLabel: 'REDIS', col: 5, row: -0.2 },
            { id: 'shap', label: 'SHAP Output', shortLabel: 'SHAP', col: 5, row: 0.2 },
        ];

        // Pipe connections (from stage index to stage index)
        this.pipes = [
            [0, 1], [0, 2],     // SimPay → CTGAN, WGAN-GP
            [1, 3], [2, 3],     // CTGAN, WGAN-GP → DuckDB
            [3, 4], [3, 5],     // DuckDB → XGBoost, Autoencoder
            [4, 6], [5, 6],     // XGBoost, AE → FastAPI
            [6, 7], [6, 8],     // FastAPI → Redis, SHAP
        ];

        // Particle state
        this.particles = [];
        this.anomalyParticle = null;
        this.time = 0;

        // Colors
        this.COL_TEAL = '#00c4cc';
        this.COL_RED = '#e5141f';
        this.COL_GREEN = '#00c853';
        this.COL_BG = '#07080d';
        this.COL_NODE_BG = 'rgba(255,255,255,0.08)';
        this.COL_NODE_BORDER = 'rgba(255,255,255,0.22)';
        this.COL_PIPE = 'rgba(255,255,255,0.12)';
    }

    /* --------------------------------------------------------
     * Get stage positions (responsive to canvas size)
     * -------------------------------------------------------- */
    _getPositions(w, h) {
        const marginX = w * 0.08;
        const marginY = h * 0.25;
        const cols = 6;
        const colW = (w - marginX * 2) / (cols - 1);
        const centerY = h * 0.5;
        const rowSpread = h * 0.18;

        return this.stages.map((s) => ({
            ...s,
            x: marginX + s.col * colW,
            y: centerY + s.row * rowSpread,
            w: Math.max(90, colW * 0.55),
            h: 38,
        }));
    }

    /* --------------------------------------------------------
     * PUBLIC: Main draw call
     * -------------------------------------------------------- */
    drawFrame(ctx, frameIndex, width, height) {
        const f = Math.max(0, Math.min(299, frameIndex));
        this.time += 0.016;

        ctx.clearRect(0, 0, width, height);

        // Background
        this._drawBackground(ctx, f, width, height);

        // Grid
        this._drawGrid(ctx, f, width, height);

        const positions = this._getPositions(width, height);

        // Pipes (phase 1+)
        this._drawPipes(ctx, f, positions, width, height);

        // Stage nodes
        this._drawNodes(ctx, f, positions, width, height);

        // Data particles (phase 2+)
        if (f > 60) {
            this._drawDataParticles(ctx, f, positions, width, height);
        }

        // Anomaly particle (phase 3)
        if (f >= 160) {
            this._drawAnomalyFlow(ctx, f, positions, width, height);
        }

        // Detection state (phase 4)
        if (f >= 240) {
            this._drawDetectedState(ctx, f, positions, width, height);
        }
    }

    /* --------------------------------------------------------
     * Background
     * -------------------------------------------------------- */
    _drawBackground(ctx, f, w, h) {
        const tealProgress = this._clamp01((f - 80) / 160);
        const r = Math.round(7 + tealProgress * 2);
        const g = Math.round(8 + tealProgress * 12);
        const b = Math.round(13 + tealProgress * 15);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(0, 0, w, h);

        // Central glow
        const grad = ctx.createRadialGradient(w * 0.5, h * 0.5, 0, w * 0.5, h * 0.5, w * 0.45);
        grad.addColorStop(0, `rgba(0, 196, 204, ${0.07 * tealProgress})`);
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);
    }

    /* --------------------------------------------------------
     * Grid
     * -------------------------------------------------------- */
    _drawGrid(ctx, f, w, h) {
        const appear = this._clamp01(f / 25);
        ctx.globalAlpha = 0.04 * appear;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 0.5;
        const sp = 50;
        for (let x = 0; x < w; x += sp) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }
        for (let y = 0; y < h; y += sp) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        }
        ctx.globalAlpha = 1;
    }

    /* --------------------------------------------------------
     * Pipes (connecting lines between stages)
     * -------------------------------------------------------- */
    _drawPipes(ctx, f, positions) {
        const pipeAppear = this._clamp01(f / 80);

        this.pipes.forEach(([fromIdx, toIdx], i) => {
            const stagger = this._clamp01(pipeAppear - i * 0.06);
            if (stagger <= 0) return;

            const from = positions[fromIdx];
            const to = positions[toIdx];

            // Draw pipe as a bezier curve
            const midX = (from.x + to.x) / 2;

            ctx.beginPath();
            ctx.moveTo(from.x + from.w / 2, from.y);
            ctx.bezierCurveTo(midX, from.y, midX, to.y, to.x - to.w / 2, to.y);
            ctx.strokeStyle = `rgba(255, 255, 255, ${0.14 * stagger})`;
            ctx.lineWidth = 2.5;
            ctx.stroke();

            // Pipe direction arrow at midpoint
            if (stagger > 0.5) {
                const arrowX = midX;
                const arrowY = (from.y + to.y) / 2;
                ctx.beginPath();
                ctx.moveTo(arrowX - 3, arrowY - 3);
                ctx.lineTo(arrowX + 3, arrowY);
                ctx.lineTo(arrowX - 3, arrowY + 3);
                ctx.strokeStyle = `rgba(255, 255, 255, ${0.16 * stagger})`;
                ctx.lineWidth = 1.2;
                ctx.stroke();
            }
        });
    }

    /* --------------------------------------------------------
     * Stage nodes (labeled rectangles)
     * -------------------------------------------------------- */
    _drawNodes(ctx, f, positions) {
        positions.forEach((pos, i) => {
            const appear = this._clamp01((f - i * 6) / 18);
            if (appear <= 0) return;

            const scale = this._easeOut(appear);
            const x = pos.x - (pos.w / 2) * scale;
            const y = pos.y - (pos.h / 2) * scale;
            const w = pos.w * scale;
            const h = pos.h * scale;
            const r = 6;

            // Glow for active nodes (when particles are flowing)
            const isActive = f > 80 + i * 8;
            if (isActive) {
                const pulse = Math.sin(this.time * 2 + i) * 0.1 + 0.9;
                const glow = ctx.createRadialGradient(
                    pos.x, pos.y, 0, pos.x, pos.y, pos.w * 0.8
                );
                glow.addColorStop(0, `rgba(0, 196, 204, ${0.12 * pulse})`);
                glow.addColorStop(1, 'rgba(0, 196, 204, 0)');
                ctx.fillStyle = glow;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, pos.w * 0.8, 0, Math.PI * 2);
                ctx.fill();
            }

            // Node background
            ctx.beginPath();
            ctx.roundRect(x, y, w, h, r);
            ctx.fillStyle = this.COL_NODE_BG;
            ctx.fill();

            // Node border
            const borderColor = isActive
                ? `rgba(0, 196, 204, ${0.55 * appear})`
                : `rgba(255, 255, 255, ${0.18 * appear})`;
            ctx.strokeStyle = borderColor;
            ctx.lineWidth = 1;
            ctx.stroke();

            // Label
            const fontSize = Math.max(8, Math.min(11, pos.w * 0.11));
            ctx.font = `500 ${fontSize}px "DM Mono", monospace`;
            ctx.fillStyle = `rgba(255, 255, 255, ${0.92 * appear})`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(pos.label, pos.x, pos.y);

            // Status indicator dot
            if (isActive) {
                ctx.beginPath();
                ctx.arc(x + 8, y + 8, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = this.COL_TEAL;
                ctx.fill();
            }
        });
    }

    /* --------------------------------------------------------
     * Data particles (teal dots flowing along pipes)
     * -------------------------------------------------------- */
    _drawDataParticles(ctx, f, positions) {
        const particleProgress = this._clamp01((f - 60) / 100);

        this.pipes.forEach(([fromIdx, toIdx], i) => {
            const stagger = this._clamp01(particleProgress - i * 0.05);
            if (stagger <= 0) return;

            const from = positions[fromIdx];
            const to = positions[toIdx];

            // Multiple particles per pipe
            for (let p = 0; p < 3; p++) {
                const base = (this.time * 0.4 + i * 0.3 + p * 0.33) % 1;
                const t = base;

                // Bezier position
                const midX = (from.x + to.x) / 2;
                const startX = from.x + from.w / 2;
                const endX = to.x - to.w / 2;

                const tt = 1 - t;
                const px = tt * tt * startX + 2 * tt * t * midX + t * t * endX;
                const py = tt * tt * from.y + 2 * tt * t * ((from.y + to.y) / 2) + t * t * to.y;

                ctx.beginPath();
                ctx.arc(px, py, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0, 196, 204, ${0.8 * stagger})`;
                ctx.fill();

                // Trail
                const trailT = ((base - 0.03) + 1) % 1;
                const trailTt = 1 - trailT;
                const tpx = trailTt * trailTt * startX + 2 * trailTt * trailT * midX + trailT * trailT * endX;
                const tpy = trailTt * trailTt * from.y + 2 * trailTt * trailT * ((from.y + to.y) / 2) + trailT * trailT * to.y;
                ctx.beginPath();
                ctx.arc(tpx, tpy, 1.5, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0, 196, 204, ${0.4 * stagger})`;
                ctx.fill();
            }
        });
    }

    /* --------------------------------------------------------
     * Anomaly particle (red, enters from left, flagged at XGBoost)
     * -------------------------------------------------------- */
    _drawAnomalyFlow(ctx, f, positions) {
        const anomalyProgress = this._clamp01((f - 160) / 80);

        // Anomaly travels along the main path: simpay → ctgan → duckdb → xgboost
        const path = [0, 1, 3, 4]; // stage indices
        const totalSegments = path.length - 1;
        const segProgress = anomalyProgress * totalSegments;
        const segIdx = Math.min(Math.floor(segProgress), totalSegments - 1);
        const segT = segProgress - segIdx;

        const from = positions[path[segIdx]];
        const to = positions[path[Math.min(segIdx + 1, path.length - 1)]];

        const ax = from.x + (to.x - from.x) * segT;
        const ay = from.y + (to.y - from.y) * segT;

        // Anomaly glow ring
        const pulseR = 12 + Math.sin(this.time * 4) * 3;
        ctx.beginPath();
        ctx.arc(ax, ay, pulseR, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(229, 20, 31, 0.08)`;
        ctx.fill();

        // Anomaly dot
        ctx.beginPath();
        ctx.arc(ax, ay, 4, 0, Math.PI * 2);
        ctx.fillStyle = this.COL_RED;
        ctx.fill();

        // Outer ring
        ctx.beginPath();
        ctx.arc(ax, ay, 8, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(229, 20, 31, ${0.4 + Math.sin(this.time * 3) * 0.2})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // "ANOMALY" label following the particle
        if (anomalyProgress > 0.1) {
            ctx.font = '600 8px "DM Mono", monospace';
            ctx.fillStyle = this.COL_RED;
            ctx.textAlign = 'center';
            ctx.fillText('ANOMALY', ax, ay - 18);
        }

        // XGBoost node highlight when anomaly reaches it
        if (anomalyProgress > 0.85) {
            const xgb = positions[4];
            const highlightAlpha = this._clamp01((anomalyProgress - 0.85) / 0.15);

            ctx.beginPath();
            ctx.roundRect(xgb.x - xgb.w / 2 - 4, xgb.y - xgb.h / 2 - 4, xgb.w + 8, xgb.h + 8, 8);
            ctx.strokeStyle = `rgba(229, 20, 31, ${0.5 * highlightAlpha})`;
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    /* --------------------------------------------------------
     * Detection confirmed state
     * -------------------------------------------------------- */
    _drawDetectedState(ctx, f, positions) {
        const progress = this._clamp01((f - 240) / 30);
        const xgb = positions[4]; // XGBoost node

        // Green containment ring
        const ringR = (xgb.w / 2 + 16) * progress;
        ctx.beginPath();
        ctx.arc(xgb.x, xgb.y, ringR, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(0, 200, 83, ${0.5 * progress})`;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]);
        ctx.stroke();
        ctx.setLineDash([]);

        // "DETECTED" label
        ctx.font = '700 10px "DM Mono", monospace';
        ctx.fillStyle = `rgba(0, 200, 83, ${progress})`;
        ctx.textAlign = 'center';
        ctx.fillText('DETECTED', xgb.x, xgb.y - xgb.h / 2 - 16);

        // Confidence score
        if (progress > 0.5) {
            const scoreAlpha = (progress - 0.5) * 2;
            ctx.font = '500 8px "DM Mono", monospace';
            ctx.fillStyle = `rgba(0, 200, 83, ${scoreAlpha * 0.7})`;
            ctx.fillText('CONFIDENCE: 99.7%', xgb.x, xgb.y + xgb.h / 2 + 18);
        }

        // Lock icon at XGBoost
        if (progress > 0.3) {
            const lockAlpha = this._clamp01((progress - 0.3) / 0.3);
            ctx.save();
            ctx.translate(xgb.x + xgb.w / 2 + 14, xgb.y);
            ctx.globalAlpha = lockAlpha;

            // Lock body
            ctx.fillStyle = 'rgba(0, 200, 83, 0.15)';
            ctx.strokeStyle = this.COL_GREEN;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.roundRect(-5, -2, 10, 8, 1.5);
            ctx.fill();
            ctx.stroke();

            // Shackle
            ctx.beginPath();
            ctx.arc(0, -2, 4, Math.PI, 0);
            ctx.stroke();

            ctx.globalAlpha = 1;
            ctx.restore();
        }

        // Scan lines
        if (progress > 0.6) {
            const scanAlpha = (progress - 0.6) * 0.04;
            const w = ctx.canvas.width / (window.devicePixelRatio || 1);
            const h = ctx.canvas.height / (window.devicePixelRatio || 1);
            for (let sy = 0; sy < h; sy += 4) {
                const wave = Math.sin(sy * 0.08 + this.time * 3) * 0.5 + 0.5;
                ctx.fillStyle = `rgba(0, 200, 83, ${scanAlpha * wave})`;
                ctx.fillRect(0, sy, w, 1);
            }
        }
    }

    /* --------------------------------------------------------
     * Utilities
     * -------------------------------------------------------- */
    _clamp01(v) { return Math.max(0, Math.min(1, v)); }
    _easeOut(t) { return 1 - Math.pow(1 - t, 3); }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = PipelineEngineRenderer;
}
