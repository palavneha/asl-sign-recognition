/* elegant-carousel.js */
const elegantSlides = [
    {
        title: 'Express Freely',
        subtitle: 'Precision ASL Translation',
        description: 'Turn your gestures into clear, accurate text instantly using our advanced AI-driven hand sign recognition.',
        accent: '#52833C', // Charity Green
        imageUrl: 'https://images.unsplash.com/photo-1544465544-1b71aee9dfa3?w=900&h=1200&fit=crop&q=80', // Sign language communication image
    },
    {
        title: 'Build Sentences',
        subtitle: 'Word Builder Tool',
        description: 'String letters together to form complete sentences, construct words dynamically and save your messages seamlessly.',
        accent: '#9A7A51', // Secondary Beige
        imageUrl: 'https://images.unsplash.com/photo-1522202176988-66273c2fd55f?w=900&h=1200&fit=crop&q=80',
    },
    {
        title: 'Visual Learning',
        subtitle: 'Interactive Dictionary',
        description: 'Master the alphabet with real fingerspelling demonstrations from authentic sources and enhance your signing skills.',
        accent: '#B4B5A3', // Card Gray
        imageUrl: 'https://images.unsplash.com/photo-1543269865-cbf427effbad?w=900&h=1200&fit=crop&q=80',
    },
    {
        title: 'Accessibility First',
        subtitle: 'Bridge the Gap',
        description: 'A modern tool designed to break down communication barriers, empowering everyone to connect and understand each other better.',
        accent: '#D4A955', // Gold / Light Beige tone
        imageUrl: 'https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=900&h=1200&fit=crop&q=80',
    }
];

class ElegantCarousel {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) return;

        this.currentIndex = 0;
        this.isTransitioning = false;
        this.progress = 0;
        this.isPaused = false;

        this.SLIDE_DURATION = 6000;
        this.TRANSITION_DURATION = 800;

        this.intervalRef = null;
        this.progressRef = null;

        this.touchStartX = 0;
        this.touchEndX = 0;

        this.initDOM();
        this.attachEventListeners();
        this.renderSlide(0);
        this.startTimers();
    }

    initDOM() {
        // Elements to update
        this.bgWash = this.container.querySelector('.carousel-bg-wash');
        this.numText = this.container.querySelector('.carousel-num-text');
        this.title = this.container.querySelector('.carousel-title');
        this.subtitle = this.container.querySelector('.carousel-subtitle');
        this.description = this.container.querySelector('.carousel-description');
        this.image = this.container.querySelector('.carousel-image');
        this.imageOverlay = this.container.querySelector('.carousel-image-overlay');
        this.cornerTl = this.container.querySelector('.carousel-frame-corner--tl');
        this.cornerBr = this.container.querySelector('.carousel-frame-corner--br');

        // Progress items
        this.progressItems = this.container.querySelectorAll('.carousel-progress-item');
        this.progressFills = this.container.querySelectorAll('.carousel-progress-fill');

        // Animated text elements
        this.animatedElements = [
            this.container.querySelector('.carousel-collection-num'),
            this.title,
            this.subtitle,
            this.description,
            this.image.parentElement // .carousel-image-frame
        ];
    }

    attachEventListeners() {
        // Navigation arrows
        const prevBtn = this.container.querySelector('#prevSlideBtn');
        const nextBtn = this.container.querySelector('#nextSlideBtn');

        if (prevBtn) prevBtn.addEventListener('click', () => this.goPrev());
        if (nextBtn) nextBtn.addEventListener('click', () => this.goNext());

        // Progress bar clicks
        this.progressItems.forEach((item, index) => {
            item.addEventListener('click', () => {
                this.goToSlide(index);
            });
        });

        // Pause on hover
        this.container.addEventListener('mouseenter', () => this.isPaused = true);
        this.container.addEventListener('mouseleave', () => this.isPaused = false);

        // Touch events
        this.container.addEventListener('touchstart', (e) => {
            this.touchStartX = e.targetTouches[0].clientX;
        }, { passive: true });

        this.container.addEventListener('touchmove', (e) => {
            this.touchEndX = e.targetTouches[0].clientX;
        }, { passive: true });

        this.container.addEventListener('touchend', () => {
            const diff = this.touchStartX - this.touchEndX;
            if (Math.abs(diff) > 60) {
                if (diff > 0) this.goNext();
                else this.goPrev();
            }
        });
    }

    startTimers() {
        if (this.intervalRef) clearInterval(this.intervalRef);
        if (this.progressRef) clearInterval(this.progressRef);

        this.progressRef = setInterval(() => {
            if (this.isPaused) return;
            this.progress += 100 / (this.SLIDE_DURATION / 50);
            if (this.progress >= 100) this.progress = 100;
            this.updateProgressBar();
        }, 50);

        this.intervalRef = setInterval(() => {
            if (!this.isPaused) this.goNext();
        }, this.SLIDE_DURATION);
    }

    goToSlide(index) {
        if (this.isTransitioning || index === this.currentIndex) return;

        this.isTransitioning = true;
        this.progress = 0;

        // Add transitioning class
        this.animatedElements.forEach(el => {
            el.classList.remove('visible');
            el.classList.add('transitioning');
        });

        this.updateProgressBar();

        setTimeout(() => {
            this.currentIndex = index;
            this.renderSlide(index);

            // Remove Ken Burns reset and re-apply
            this.image.classList.remove('ken-burns');

            setTimeout(() => {
                this.animatedElements.forEach(el => {
                    el.classList.remove('transitioning');
                    el.classList.add('visible');
                });

                // Trigger Ken burns
                // Using offsetWidth triggers reflow
                void this.image.offsetWidth;
                this.image.classList.add('ken-burns');

                this.isTransitioning = false;
            }, 50);
        }, this.TRANSITION_DURATION / 2);
    }

    goNext() {
        const nextIndex = (this.currentIndex + 1) % elegantSlides.length;
        this.goToSlide(nextIndex);
    }

    goPrev() {
        const prevIndex = (this.currentIndex - 1 + elegantSlides.length) % elegantSlides.length;
        this.goToSlide(prevIndex);
    }

    renderSlide(index) {
        const slide = elegantSlides[index];

        // Update texts
        this.numText.textContent = `${String(index + 1).padStart(2, '0')} / ${String(elegantSlides.length).padStart(2, '0')}`;
        this.title.textContent = slide.title;
        this.subtitle.textContent = slide.subtitle;
        this.subtitle.style.color = slide.accent;
        this.description.textContent = slide.description;

        // Update image
        this.image.src = slide.imageUrl;
        this.image.alt = slide.title;

        // Update accents
        this.bgWash.style.background = `radial-gradient(ellipse at 70% 50%, ${slide.accent}18 0%, transparent 70%)`;
        this.imageOverlay.style.background = `linear-gradient(135deg, ${slide.accent}22 0%, transparent 70%)`;
        this.cornerTl.style.borderColor = slide.accent;
        this.cornerBr.style.borderColor = slide.accent;

        this.updateProgressBar();
    }

    updateProgressBar() {
        const slide = elegantSlides[this.currentIndex];

        this.progressItems.forEach((item, index) => {
            if (index === this.currentIndex) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        this.progressFills.forEach((fill, index) => {
            if (index === this.currentIndex) {
                fill.style.width = `${this.progress}%`;
                fill.style.backgroundColor = slide.accent;
            } else if (index < this.currentIndex) {
                fill.style.width = '100%';
                fill.style.backgroundColor = 'rgba(255,255,255,0.2)';
            } else {
                fill.style.width = '0%';
                fill.style.backgroundColor = 'transparent';
            }
        });
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    new ElegantCarousel('elegantCarousel');
});
