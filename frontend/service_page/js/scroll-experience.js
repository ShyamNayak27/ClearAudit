class ScrollExperience {
    constructor(threatEngine) {
        this.engine = threatEngine;
        this.initLenis();
        this.initScrollTrigger();
        this.initAnimations();
    }

    initLenis() {
        // Initialize high-performance smooth scrolling
        this.lenis = new Lenis({
            duration: 1.2,
            easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
            direction: 'vertical',
            gestureDirection: 'vertical',
            smooth: true,
            smoothTouch: false, // Don't hijack native mobile touch scrolling
        });

        const raf = (time) => {
            this.lenis.raf(time);
            requestAnimationFrame(raf);
        };
        requestAnimationFrame(raf);
    }

    initScrollTrigger() {
        gsap.registerPlugin(ScrollTrigger);

        if (!this.engine) return;

        // Ensure Lenis and GSAP stay synced
        this.lenis.on('scroll', ScrollTrigger.update);
        gsap.ticker.add((time) => {
            this.lenis.raf(time * 1000);
        });
        gsap.ticker.lagSmoothing(0);

        ScrollTrigger.create({
            trigger: '.hero-scroll-container',
            start: 'top top',
            end: 'bottom bottom',
            scrub: 0.1, // Slight smoothing on the scrub
            onUpdate: (self) => {
                // Determine exact frame based on scroll progress
                const frameIdx = Math.floor(self.progress * (this.engine.totalFrames - 1));
                this.engine.setFrame(frameIdx);

                // Fade in early
                const stickyView = document.querySelector('.hero-sticky-view');
                if (stickyView) {
                    // Fade in completely by 15% scroll
                    const opacity = Math.min(1, self.progress * 6.5);
                    stickyView.style.opacity = opacity;
                }
            }
        });
    }

    initAnimations() {
        // Standard fade-up reveal for content sections
        gsap.utils.toArray('.reveal').forEach(el => {
            let delay = 0;
            if (el.classList.contains('reveal-d1')) delay = 0.15;
            if (el.classList.contains('reveal-d2')) delay = 0.3;
            if (el.classList.contains('reveal-d3')) delay = 0.45;

            gsap.fromTo(el,
                { opacity: 0, y: 40 },
                {
                    opacity: 1,
                    y: 0,
                    duration: 1,
                    delay: delay,
                    ease: "power3.out",
                    scrollTrigger: {
                        trigger: el,
                        start: "top 85%",
                        toggleActions: "play none none reverse"
                    }
                }
            );
        });

        // Number Counter Animation for elements with data-count
        gsap.utils.toArray('.metric-number').forEach(el => {
            const finalVal = parseFloat(el.getAttribute('data-count'));
            const suffix = el.getAttribute('data-suffix') || '';
            const decimals = parseInt(el.getAttribute('data-decimals') || 0);

            if (isNaN(finalVal)) return;

            gsap.fromTo(el,
                { innerHTML: 0 },
                {
                    innerHTML: finalVal,
                    duration: 2,
                    ease: "power2.out",
                    scrollTrigger: {
                        trigger: el,
                        start: "top 85%",
                        toggleActions: "play none none reverse"
                    },
                    snap: { innerHTML: Math.pow(10, -decimals) },
                    onUpdate: function () {
                        const val = parseFloat(this.targets()[0].innerHTML);
                        el.innerHTML = val.toFixed(decimals) + suffix;
                    }
                }
            );
        });
    }
}

window.ScrollExperience = ScrollExperience;
