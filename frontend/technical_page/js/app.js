/**
 * App.js — Technical Page Entry Point
 * Wires PipelineEngineRenderer + ScrollExperience,
 * section reveals, animated metric counters, and static architecture canvas.
 */
document.addEventListener('DOMContentLoaded', () => {

    // ============================================================
    // 1. HERO — Pipeline scrollytelling
    // ============================================================
    const renderer = new PipelineEngineRenderer();
    const scrollExp = new ScrollExperience(renderer);

    // ============================================================
    // 2. SECTION REVEALS
    // ============================================================
    const reveals = document.querySelectorAll('.reveal');
    const revealObs = new IntersectionObserver(
        (entries) => {
            entries.forEach((e) => {
                if (e.isIntersecting) {
                    e.target.classList.add('visible');
                    revealObs.unobserve(e.target);
                }
            });
        },
        { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
    );
    reveals.forEach((el) => revealObs.observe(el));

    // ============================================================
    // 3. ANIMATED METRIC COUNTERS
    // ============================================================
    const counters = document.querySelectorAll('[data-count]');
    const counterObs = new IntersectionObserver(
        (entries) => {
            entries.forEach((e) => {
                if (e.isIntersecting) {
                    animateCounter(e.target);
                    counterObs.unobserve(e.target);
                }
            });
        },
        { threshold: 0.5 }
    );
    counters.forEach((el) => counterObs.observe(el));

    function animateCounter(el) {
        const target = parseFloat(el.dataset.count);
        const suffix = el.dataset.suffix || '';
        const prefix = el.dataset.prefix || '';
        const decimals = (el.dataset.decimals || 0) | 0;
        const duration = 2200;
        const start = performance.now();

        function tick(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = prefix + (target * eased).toFixed(decimals) + suffix;
            if (progress < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    // ============================================================
    // 4. STATIC ARCHITECTURE DIAGRAM CANVAS
    // ============================================================
    const archCanvas = document.getElementById('arch-canvas');
    if (archCanvas) {
        const ctx = archCanvas.getContext('2d');
        const dpr = Math.min(window.devicePixelRatio || 1, 2);

        function drawArch() {
            const rect = archCanvas.parentElement.getBoundingClientRect();
            const w = rect.width - 64; // account for padding
            const h = 200;
            archCanvas.width = w * dpr;
            archCanvas.height = h * dpr;
            archCanvas.style.width = w + 'px';
            archCanvas.style.height = h + 'px';
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

            // Draw a simplified architecture flow
            const stages = [
                'SimPay VN', 'CTGAN', 'WGAN-GP', 'DuckDB', 'XGBoost', 'AE', 'FastAPI', 'Redis', 'SHAP'
            ];

            const nodeW = Math.min(85, (w - 40) / stages.length - 8);
            const nodeH = 32;
            const startX = (w - (stages.length * (nodeW + 10) - 10)) / 2;
            const cy = h / 2;

            stages.forEach((label, i) => {
                const nx = startX + i * (nodeW + 10);

                // Connector to next
                if (i < stages.length - 1) {
                    const nextX = startX + (i + 1) * (nodeW + 10);
                    ctx.beginPath();
                    ctx.moveTo(nx + nodeW, cy);
                    ctx.lineTo(nextX, cy);
                    ctx.strokeStyle = 'rgba(0, 196, 204, 0.2)';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();

                    // Arrow
                    ctx.beginPath();
                    ctx.moveTo(nextX - 4, cy - 3);
                    ctx.lineTo(nextX, cy);
                    ctx.lineTo(nextX - 4, cy + 3);
                    ctx.strokeStyle = 'rgba(0, 196, 204, 0.3)';
                    ctx.stroke();
                }

                // Node
                ctx.beginPath();
                ctx.roundRect(nx, cy - nodeH / 2, nodeW, nodeH, 4);
                ctx.fillStyle = 'rgba(0, 196, 204, 0.06)';
                ctx.fill();
                ctx.strokeStyle = 'rgba(0, 196, 204, 0.25)';
                ctx.lineWidth = 1;
                ctx.stroke();

                // Label
                const fontSize = Math.max(7, Math.min(9, nodeW * 0.11));
                ctx.font = `500 ${fontSize}px "DM Mono", monospace`;
                ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(label, nx + nodeW / 2, cy);

                // Status dot
                ctx.beginPath();
                ctx.arc(nx + 6, cy - nodeH / 2 + 6, 2, 0, Math.PI * 2);
                ctx.fillStyle = '#00c4cc';
                ctx.fill();
            });
        }

        const archObs = new IntersectionObserver(
            (entries) => {
                entries.forEach((e) => {
                    if (e.isIntersecting) {
                        drawArch();
                        archObs.unobserve(e.target);
                    }
                });
            },
            { threshold: 0.2 }
        );
        archObs.observe(archCanvas);

        let archTimer;
        window.addEventListener('resize', () => {
            clearTimeout(archTimer);
            archTimer = setTimeout(drawArch, 200);
        });
    }
});
