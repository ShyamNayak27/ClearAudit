/**
 * ScrollExperience — Idea3 HR
 * GSAP ScrollTrigger + Lenis for pipeline scrollytelling.
 * Drives PipelineEngineRenderer across dual canvases.
 */
class ScrollExperience {
    constructor(renderer) {
        this.renderer = renderer;
        this.currentFrame = 0;
        this.totalFrames = 300;

        this.fgCanvas = document.getElementById('pipeline-canvas');
        this.bgCanvas = document.getElementById('ambient-canvas');
        this.fgCtx = this.fgCanvas.getContext('2d');
        this.bgCtx = this.bgCanvas.getContext('2d');

        this.heroEyebrow = document.querySelector('.hero-eyebrow');
        this.heroTitle = document.querySelector('.hero-title');
        this.heroSubtitle = document.querySelector('.hero-subtitle');
        this.phaseLabel = document.querySelector('.phase-label');
        this.scrollCue = document.querySelector('.scroll-cue');

        this.lenis = null;
        this._onResize = this._debounce(this._handleResize.bind(this), 150);
        this._init();
    }

    _init() {
        this._sizeCanvases();
        window.addEventListener('resize', this._onResize);

        // Lenis
        this.lenis = new Lenis({
            duration: 1.2,
            easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
            orientation: 'vertical',
            gestureOrientation: 'vertical',
            smoothWheel: true,
        });

        this.lenis.on('scroll', ScrollTrigger.update);
        gsap.ticker.add((time) => { this.lenis.raf(time * 1000); });
        gsap.ticker.lagSmoothing(0);

        this._setupScrollTrigger();

        requestAnimationFrame(() => {
            if (this.heroEyebrow) this.heroEyebrow.classList.add('visible');
            if (this.heroTitle) this.heroTitle.classList.add('visible');
            if (this.heroSubtitle) this.heroSubtitle.classList.add('visible');
        });

        this._drawCurrentFrame();
    }

    _setupScrollTrigger() {
        const self = this;
        ScrollTrigger.create({
            trigger: '#hero',
            start: 'top top',
            end: 'bottom bottom',
            scrub: 0.5,
            onUpdate: (st) => {
                self.currentFrame = Math.round(st.progress * (self.totalFrames - 1));
                self._drawCurrentFrame();
                self._updatePhaseLabel();
                self._updateHeroText(st.progress);
                self._updateScrollCue(st.progress);
            },
        });
    }

    _drawCurrentFrame() {
        const fgW = this.fgCanvas.width / (Math.min(window.devicePixelRatio || 1, 2));
        const fgH = this.fgCanvas.height / (Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.drawFrame(this.fgCtx, this.currentFrame, fgW, fgH);

        const bgW = this.bgCanvas.width / (Math.min(window.devicePixelRatio || 1, 2));
        const bgH = this.bgCanvas.height / (Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.drawFrame(this.bgCtx, this.currentFrame, bgW, bgH);
    }

    _updatePhaseLabel() {
        if (!this.phaseLabel) return;
        const f = this.currentFrame;
        let text = '';
        if (f < 80) text = 'STAGE 01 / PIPELINE INIT';
        else if (f < 160) text = 'STAGE 02 / DATA FLOW';
        else if (f < 240) text = 'STAGE 03 / ANOMALY DETECTION';
        else text = 'STAGE 04 / FRAUD CONTAINED';
        this.phaseLabel.textContent = text;
    }

    _updateHeroText(progress) {
        const fadeStart = 0.04;
        const fadeEnd = 0.14;
        if (progress > fadeStart) {
            const opacity = 1 - this._clamp01((progress - fadeStart) / (fadeEnd - fadeStart));
            if (this.heroEyebrow) this.heroEyebrow.style.opacity = opacity;
            if (this.heroTitle) this.heroTitle.style.opacity = opacity;
            if (this.heroSubtitle) this.heroSubtitle.style.opacity = opacity;
        }
    }

    _updateScrollCue(progress) {
        if (!this.scrollCue) return;
        this.scrollCue.style.opacity = progress < 0.02 ? '0.35' : '0';
    }

    _sizeCanvases() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = window.innerWidth;
        const h = window.innerHeight;

        this.fgCanvas.width = w * dpr;
        this.fgCanvas.height = h * dpr;
        this.fgCanvas.style.width = w + 'px';
        this.fgCanvas.style.height = h + 'px';
        this.fgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const bgScale = 1.1;
        this.bgCanvas.width = w * bgScale * dpr;
        this.bgCanvas.height = h * bgScale * dpr;
        this.bgCtx.setTransform(dpr * bgScale, 0, 0, dpr * bgScale, 0, 0);
    }

    _handleResize() {
        this._sizeCanvases();
        this._drawCurrentFrame();
    }

    _debounce(fn, ms) {
        let timer;
        return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
    }

    _clamp01(v) { return Math.max(0, Math.min(1, v)); }

    destroy() {
        window.removeEventListener('resize', this._onResize);
        ScrollTrigger.getAll().forEach(st => st.kill());
        if (this.lenis) this.lenis.destroy();
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = ScrollExperience;
}
