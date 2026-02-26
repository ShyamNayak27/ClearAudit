// ─────────────────────────────────────────────
// CURSOR
// ─────────────────────────────────────────────
const cursor = document.getElementById('cursor');
const cursorRing = document.getElementById('cursor-ring');
let mx = -100, my = -100, rx = -100, ry = -100;

document.addEventListener('mousemove', e => {
    mx = e.clientX;
    my = e.clientY;
});

(function cursorLoop() {
    rx += (mx - rx) * 0.14;
    ry += (my - ry) * 0.14;
    cursor.style.left = mx + 'px';
    cursor.style.top = my + 'px';
    cursorRing.style.left = rx + 'px';
    cursorRing.style.top = ry + 'px';
    requestAnimationFrame(cursorLoop);
})();

// ─────────────────────────────────────────────
// PARTICLE ENGINE
// ─────────────────────────────────────────────
class ParticleEngine {
    constructor() {
        this.canvas = document.getElementById('particle-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.W = 0; this.H = 0;
        this.particles = [];
        this.splitProgress = 0; // 0 = single stream, 1 = fully split, -1 = merged
        this.mergeProgress = 0; // 0..1 for reconvergence
        this.hoverSide = null; // 'business' | 'engineering' | null
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.spawnParticles();
        this.raf();
    }

    resize() {
        this.W = this.canvas.width = window.innerWidth;
        this.H = this.canvas.height = window.innerHeight;
    }

    spawnParticle(forceSingle = false) {
        const sp = this.splitProgress;
        const mp = this.mergeProgress;
        const cx = this.W / 2;

        let side = 'single';
        if (sp > 0.05 && !forceSingle) {
            const r = Math.random();
            side = r < 0.5 ? 'business' : 'engineering';
        }

        const isEngChars = side === 'engineering' && Math.random() > 0.45;
        const chars = ['0', '1', 'F', 'X', '∑', 'λ', '∇', '⚑'];

        const spread = 18;
        return {
            x: cx + (Math.random() - 0.5) * spread * (sp < 0.1 ? 1 : 0.3),
            y: this.H + 10 + Math.random() * 40,
            side,
            speed: 1.4 + Math.random() * 2.2,
            size: 1.5 + Math.random() * 2,
            opacity: 0.6 + Math.random() * 0.4,
            char: isEngChars ? chars[Math.floor(Math.random() * chars.length)] : null,
            charOpacity: 0.7 + Math.random() * 0.3,
            life: 0,
            maxLife: this.H * 1.3,
            wobble: (Math.random() - 0.5) * 0.3,
        };
    }

    spawnParticles() {
        this.particles = [];
        for (let i = 0; i < 70; i++) {
            const p = this.spawnParticle(true);
            p.y = Math.random() * window.innerHeight;
            p.life = Math.random() * p.maxLife * 0.5;
            this.particles.push(p);
        }
    }

    getTargetX(p) {
        const cx = this.W / 2;
        const sp = Math.max(0, this.splitProgress);
        const mp = this.mergeProgress;
        const maxSplit = Math.min(this.W * 0.32, 340);

        if (mp > 0) {
            const splitX = p.side === 'business'
                ? cx - maxSplit * (1 - mp)
                : p.side === 'engineering'
                    ? cx + maxSplit * (1 - mp)
                    : cx;
            return splitX + p.wobble * 12;
        }

        if (sp < 0.05 || p.side === 'single') {
            return cx + p.wobble * 10;
        }
        const offset = maxSplit * sp;
        if (p.side === 'business') return cx - offset + p.wobble * 12;
        if (p.side === 'engineering') return cx + offset + p.wobble * 12;
        return cx;
    }

    draw() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.W, this.H);

        if (this.particles.length < 80) {
            this.particles.push(this.spawnParticle());
        }
        if (Math.random() < 0.35) this.particles.push(this.spawnParticle());

        const businessDim = this.hoverSide === 'engineering' ? 0.25 : 1;
        const engDim = this.hoverSide === 'business' ? 0.25 : 1;

        this.particles = this.particles.filter(p => {
            p.life += p.speed;
            p.y -= p.speed;

            const tx = this.getTargetX(p);
            p.x += (tx - p.x) * 0.04;

            if (p.y < -20 || p.life > p.maxLife) return false;

            const fadeBottom = Math.min(1, (this.H - p.y) / 80);
            const fadeTop = Math.min(1, p.y / 80);
            let alpha = p.opacity * fadeBottom * fadeTop;

            if (p.side === 'business') alpha *= businessDim;
            if (p.side === 'engineering') alpha *= engDim;

            ctx.save();
            ctx.globalAlpha = alpha;

            if (p.char) {
                ctx.font = `${8 + p.size}px "DM Mono", monospace`;
                ctx.fillStyle = p.side === 'engineering' ? '#6b9aff' : '#00c4cc';
                ctx.fillText(p.char, p.x, p.y);
            } else {
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = p.side === 'engineering' ? '#6b9aff' : '#00c4cc';
                ctx.shadowColor = p.side === 'engineering' ? '#6b9aff' : '#00c4cc';
                ctx.shadowBlur = 8;
                ctx.fill();
            }

            ctx.restore();
            return true;
        });

        if (this.splitProgress > 0.05) {
            const cx = this.W / 2;
            const sp = Math.min(1, this.splitProgress);
            const mp = this.mergeProgress;
            const maxSplit = Math.min(this.W * 0.32, 340);

            const drawTrail = (x, color, dim) => {
                if (dim < 0.1) return;
                const grad = ctx.createLinearGradient(x, this.H, x, 0);
                grad.addColorStop(0, 'transparent');
                grad.addColorStop(0.3, color.replace('1)', `${0.06 * dim})`));
                grad.addColorStop(0.7, color.replace('1)', `${0.04 * dim})`));
                grad.addColorStop(1, 'transparent');
                ctx.beginPath();
                ctx.moveTo(x, this.H);
                ctx.lineTo(x, 0);
                ctx.strokeStyle = grad;
                ctx.lineWidth = 1;
                ctx.globalAlpha = 1;
                ctx.stroke();
            };

            const bx = mp > 0 ? cx - maxSplit * (1 - mp) : cx - maxSplit * sp;
            const ex = mp > 0 ? cx + maxSplit * (1 - mp) : cx + maxSplit * sp;

            drawTrail(bx, 'rgba(0,196,204,1)', businessDim * sp * (1 - mp * 0.8));
            drawTrail(ex, 'rgba(107,154,255,1)', engDim * sp * (1 - mp * 0.8));

            if (mp > 0.4) {
                drawTrail(cx, 'rgba(0,196,204,1)', mp * 0.6);
            }
        }
    }

    raf() {
        this.draw();
        requestAnimationFrame(() => this.raf());
    }
}

const engine = new ParticleEngine();

// ─────────────────────────────────────────────
// SCROLL LOGIC
// ─────────────────────────────────────────────
gsap.registerPlugin(ScrollTrigger);

// Zone 1 entrance
gsap.timeline({ defaults: { ease: 'power3.out' } })
    .to('#wordmark', { opacity: 1, y: 0, duration: 1.1, delay: 0.3 })
    .to('#tagline', { opacity: 1, duration: 0.8 }, '-=0.4')
    .to('#scroll-hint', { opacity: 1, duration: 0.6 }, '-=0.2');

// Zone 2 — split cards appear
ScrollTrigger.create({
    trigger: '#zone-split',
    start: 'top 80%',
    onEnter: () => {
        gsap.to('#split-layout', { opacity: 1, duration: 0.7, ease: 'power2.out' });
        gsap.to('.metric-pill', {
            opacity: 1,
            y: 0,
            duration: 0.5,
            stagger: 0.12,
            delay: 0.3,
            ease: 'power2.out'
        });
    }
});

// Scroll-driven split progress
ScrollTrigger.create({
    trigger: '#zone-split',
    start: 'top bottom',
    end: 'bottom top',
    scrub: true,
    onUpdate: (self) => {
        engine.splitProgress = Math.min(1, self.progress * 2.5);
    }
});

// Merge — zone 3/4
ScrollTrigger.create({
    trigger: '#zone-merge',
    start: 'top 80%',
    onEnter: () => {
        gsap.to('#merge-label', { opacity: 0.4, duration: 0.8 });
    }
});

ScrollTrigger.create({
    trigger: '#zone-merge',
    start: 'top bottom',
    end: 'bottom top',
    scrub: true,
    onUpdate: (self) => {
        engine.mergeProgress = Math.min(1, self.progress * 2);
    }
});

// Creator section
ScrollTrigger.create({
    trigger: '#zone-creator',
    start: 'top 75%',
    onEnter: () => {
        gsap.timeline({ defaults: { ease: 'power3.out' } })
            .to('#creator-eyebrow', { opacity: 1, duration: 0.6 })
            .to('#creator-name', { opacity: 1, y: 0, duration: 0.9 }, '-=0.3')
            .to('#creator-bio', { opacity: 1, duration: 0.7 }, '-=0.4')
            .to('#creator-availability', { opacity: 1, duration: 0.5 }, '-=0.3')
            .to('#creator-links', { opacity: 1, duration: 0.5 }, '-=0.2');
    }
});

// ─────────────────────────────────────────────
// CARD HOVER
// ─────────────────────────────────────────────
const splitLayout = document.getElementById('split-layout');
document.getElementById('card-business').addEventListener('mouseenter', () => {
    engine.hoverSide = 'business';
    splitLayout.classList.add('hover-business');
    splitLayout.classList.remove('hover-engineering');
});
document.getElementById('card-engineering').addEventListener('mouseenter', () => {
    engine.hoverSide = 'engineering';
    splitLayout.classList.add('hover-engineering');
    splitLayout.classList.remove('hover-business');
});
document.getElementById('card-business').addEventListener('mouseleave', () => {
    engine.hoverSide = null;
    splitLayout.classList.remove('hover-business');
});
document.getElementById('card-engineering').addEventListener('mouseleave', () => {
    engine.hoverSide = null;
    splitLayout.classList.remove('hover-engineering');
});

// ─────────────────────────────────────────────
// CLICK — flash animation then redirect
// ─────────────────────────────────────────────
function handleCardClick(e) {
    const url = e.currentTarget.getAttribute('href');
    e.preventDefault();
    const flash = document.getElementById('flash-overlay');

    const origSpeeds = engine.particles.map(p => p.speed);
    engine.particles.forEach(p => { p.speed *= 3.5; });

    gsap.timeline()
        .to(flash, {
            opacity: 0.85,
            duration: 0.55,
            ease: 'power2.in',
            onComplete: () => {
                window.location.href = url;
            }
        })
        .to(flash, { opacity: 0, duration: 0.4, ease: 'power2.out', delay: 0.05 });
}

document.getElementById('card-business').addEventListener('click', handleCardClick);
document.getElementById('card-engineering').addEventListener('click', handleCardClick);
