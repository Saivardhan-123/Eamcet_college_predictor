// ===========================
// FUTURISTIC UI JAVASCRIPT
// ===========================

// Page Transition System
class PageTransition {
    constructor() {
        this.transitionOverlay = null;
        this.init();
    }

    init() {
        this.createTransitionOverlay();
        this.setupPageTransitions();
        this.setupMobileMenu();
        this.setupParticleBackground();
        this.setupAnimations();
    }

    createTransitionOverlay() {
        this.transitionOverlay = document.createElement('div');
        this.transitionOverlay.className = 'page-transition';
        this.transitionOverlay.innerHTML = '<div class="transition-loader"></div>';
        document.body.appendChild(this.transitionOverlay);
    }

    setupPageTransitions() {
        // Intercept navigation links
        document.addEventListener('click', (e) => {
            const link = e.target.closest('a');
            if (link && link.href && !link.href.includes('#') && 
                link.href.includes(window.location.origin) &&
                !link.hasAttribute('target')) {
                
                e.preventDefault();
                this.navigateToPage(link.href);
            }
        });

        // Handle form submissions
        document.addEventListener('submit', (e) => {
            const form = e.target;
            if (form && form.method === 'post') {
                this.showTransition();
                // Let form submit naturally, but show transition
                setTimeout(() => {
                    // Form will handle the actual navigation
                }, 100);
            }
        });
    }

    navigateToPage(url) {
        this.showTransition();
        
        setTimeout(() => {
            window.location.href = url;
        }, 300);
    }

    showTransition() {
        this.transitionOverlay.classList.add('active');
    }

    hideTransition() {
        this.transitionOverlay.classList.remove('active');
    }

    setupMobileMenu() {
        const mobileMenuBtn = document.getElementById('mobile-menu-btn');
        const navMenu = document.getElementById('nav-menu');

        if (mobileMenuBtn && navMenu) {
            mobileMenuBtn.addEventListener('click', () => {
                navMenu.classList.toggle('active');
                
                // Animate hamburger icon
                const icon = mobileMenuBtn.textContent;
                mobileMenuBtn.textContent = icon === '☰' ? '✕' : '☰';
            });

            // Close menu when clicking outside
            document.addEventListener('click', (e) => {
                if (!navMenu.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
                    navMenu.classList.remove('active');
                    mobileMenuBtn.textContent = '☰';
                }
            });
        }
    }

    setupParticleBackground() {
        // Create particle canvas
        const canvas = document.createElement('canvas');
        canvas.id = 'particle-canvas';
        document.body.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        let particles = [];
        
        // Resize canvas
        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);
        
        // Particle class
        class Particle {
            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.vx = (Math.random() - 0.5) * 0.5;
                this.vy = (Math.random() - 0.5) * 0.5;
                this.radius = Math.random() * 2 + 1;
                this.opacity = Math.random() * 0.5 + 0.2;
                this.hue = Math.random() * 60 + 200; // Blue to cyan range
            }
            
            update() {
                this.x += this.vx;
                this.y += this.vy;
                
                // Wrap around screen
                if (this.x < 0) this.x = canvas.width;
                if (this.x > canvas.width) this.x = 0;
                if (this.y < 0) this.y = canvas.height;
                if (this.y > canvas.height) this.y = 0;
            }
            
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${this.hue}, 100%, 70%, ${this.opacity})`;
                ctx.fill();
                
                // Add glow effect
                ctx.shadowBlur = 10;
                ctx.shadowColor = `hsl(${this.hue}, 100%, 70%)`;
            }
        }
        
        // Create particles
        function createParticles() {
            const particleCount = Math.min(Math.floor((canvas.width * canvas.height) / 15000), 100);
            particles = [];
            
            for (let i = 0; i < particleCount; i++) {
                particles.push(new Particle());
            }
        }
        
        createParticles();
        window.addEventListener('resize', createParticles);
        
        // Animation loop
        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            particles.forEach(particle => {
                particle.update();
                particle.draw();
            });
            
            // Draw connections between nearby particles
            particles.forEach((particle, i) => {
                particles.slice(i + 1).forEach(otherParticle => {
                    const dx = particle.x - otherParticle.x;
                    const dy = particle.y - otherParticle.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    if (distance < 120) {
                        ctx.beginPath();
                        ctx.moveTo(particle.x, particle.y);
                        ctx.lineTo(otherParticle.x, otherParticle.y);
                        ctx.strokeStyle = `rgba(0, 212, 255, ${0.1 * (1 - distance / 120)})`;
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }
                });
            });
            
            requestAnimationFrame(animate);
        }
        
        animate();
    }

    setupAnimations() {
        // Intersection Observer for scroll animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, observerOptions);

        // Observe all content cards
        document.querySelectorAll('.content-card, .feature-card').forEach(card => {
            observer.observe(card);
        });

        // Add floating animation to feature icons
        document.querySelectorAll('.feature-icon').forEach((icon, index) => {
            icon.style.animationDelay = `${index * 0.2}s`;
            icon.classList.add('floating');
        });

        // Add pulse effect to CTA buttons
        document.querySelectorAll('.btn-primary, .btn-accent').forEach(btn => {
            btn.addEventListener('mouseenter', () => {
                btn.classList.add('pulse');
            });
            
            btn.addEventListener('mouseleave', () => {
                btn.classList.remove('pulse');
            });
        });
    }
}

// Form Enhancement
class FormEnhancer {
    constructor() {
        this.init();
    }

    init() {
        this.enhanceInputs();
        this.addLoadingStates();
    }

    enhanceInputs() {
        // Add focus animations to form inputs
        document.querySelectorAll('.form-input, .form-select').forEach(input => {
            const group = input.closest('.form-group');
            
            input.addEventListener('focus', () => {
                group?.classList.add('focused');
            });
            
            input.addEventListener('blur', () => {
                group?.classList.remove('focused');
            });

            // Add ripple effect
            input.addEventListener('click', (e) => {
                this.createRipple(e, input);
            });
        });
    }

    createRipple(event, element) {
        const rect = element.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const ripple = document.createElement('div');
        ripple.style.position = 'absolute';
        ripple.style.borderRadius = '50%';
        ripple.style.background = 'rgba(0, 212, 255, 0.3)';
        ripple.style.transform = 'scale(0)';
        ripple.style.animation = 'ripple 0.6s linear';
        ripple.style.left = `${x - 10}px`;
        ripple.style.top = `${y - 10}px`;
        ripple.style.width = '20px';
        ripple.style.height = '20px';
        ripple.style.pointerEvents = 'none';

        element.style.position = 'relative';
        element.appendChild(ripple);

        setTimeout(() => {
            ripple.remove();
        }, 600);
    }

    addLoadingStates() {
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', (e) => {
                const submitBtn = form.querySelector('input[type="submit"], button[type="submit"]');
                if (submitBtn) {
                    const originalText = submitBtn.value || submitBtn.textContent;
                    submitBtn.disabled = true;
                    
                    if (submitBtn.tagName === 'INPUT') {
                        submitBtn.value = 'Processing...';
                    } else {
                        submitBtn.innerHTML = '<div class="loading-spinner"></div> Processing...';
                    }

                    // Re-enable after 10 seconds as fallback
                    setTimeout(() => {
                        submitBtn.disabled = false;
                        if (submitBtn.tagName === 'INPUT') {
                            submitBtn.value = originalText;
                        } else {
                            submitBtn.textContent = originalText;
                        }
                    }, 10000);
                }
            });
        });
    }
}

// Table Enhancements
class TableEnhancer {
    constructor() {
        this.init();
    }

    init() {
        this.enhanceTables();
    }

    enhanceTables() {
        document.querySelectorAll('table').forEach(table => {
            // Add data-table class for styling
            table.classList.add('data-table');
            
            // Add hover effects to rows
            table.querySelectorAll('tbody tr').forEach((row, index) => {
                row.style.animationDelay = `${index * 0.05}s`;
                
                row.addEventListener('mouseenter', () => {
                    row.style.transform = 'scale(1.01)';
                    row.style.zIndex = '10';
                });
                
                row.addEventListener('mouseleave', () => {
                    row.style.transform = 'scale(1)';
                    row.style.zIndex = '1';
                });
            });
        });
    }
}

// Notification System
class NotificationSystem {
    constructor() {
        this.container = null;
        this.init();
    }

    init() {
        this.createContainer();
    }

    createContainer() {
        this.container = document.createElement('div');
        this.container.className = 'notification-container';
        this.container.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 10px;
        `;
        document.body.appendChild(this.container);
    }

    show(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type}`;
        notification.style.cssText = `
            max-width: 400px;
            transform: translateX(100%);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        `;
        notification.textContent = message;

        this.container.appendChild(notification);

        // Animate in
        requestAnimationFrame(() => {
            notification.style.transform = 'translateX(0)';
        });

        // Auto remove
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                notification.remove();
            }, 400);
        }, duration);

        // Click to dismiss
        notification.addEventListener('click', () => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                notification.remove();
            }, 400);
        });
    }
}

// Add ripple animation keyframes
const rippleStyles = document.createElement('style');
rippleStyles.textContent = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .focused .form-label {
        color: var(--neon-blue) !important;
        text-shadow: 0 0 5px var(--neon-blue);
    }
`;
document.head.appendChild(rippleStyles);

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize all systems
    window.pageTransition = new PageTransition();
    window.formEnhancer = new FormEnhancer();
    window.tableEnhancer = new TableEnhancer();
    window.notificationSystem = new NotificationSystem();

    // Hide page transition overlay
    setTimeout(() => {
        window.pageTransition.hideTransition();
    }, 500);

    // Add entrance animation to page content
    document.querySelectorAll('.content-card').forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1 + 0.3}s`;
        card.classList.add('animate-in');
    });

    // Show welcome notification on home page
    if (window.location.pathname === '/' || window.location.pathname.includes('home')) {
        setTimeout(() => {
            window.notificationSystem.show(
                'Welcome to the futuristic EAMCET Predictor! 🚀',
                'info',
                3000
            );
        }, 1000);
    }
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden
        document.title = '🎓 Come back to EAMCET Predictor';
    } else {
        // Page is visible
        document.title = document.title.replace('🎓 Come back to ', '');
    }
});

// Smooth scrolling for anchor links
document.addEventListener('click', (e) => {
    const link = e.target.closest('a[href^="#"]');
    if (link) {
        e.preventDefault();
        const target = document.querySelector(link.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }
});

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K for search (if search exists)
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('input[type="search"], input[placeholder*="search"]');
        if (searchInput) {
            searchInput.focus();
        }
    }
    
    // Escape to close mobile menu
    if (e.key === 'Escape') {
        const navMenu = document.getElementById('nav-menu');
        const mobileMenuBtn = document.getElementById('mobile-menu-btn');
        if (navMenu?.classList.contains('active')) {
            navMenu.classList.remove('active');
            if (mobileMenuBtn) mobileMenuBtn.textContent = '☰';
        }
    }
});
