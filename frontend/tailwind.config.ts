/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      // ONI Brand Colors
      colors: {
        // Primary ONI brand colors
        oni: {
          primary: '#6366f1',
          secondary: '#8b5cf6',
          accent: '#10b981',
          warning: '#f59e0b',
          danger: '#ef4444',
          dark: '#0f0f23',
          'dark-secondary': '#1a1a3e',
          'dark-tertiary': '#2d1b69',
        },
        
        // Blockchain/Crypto theme
        crypto: {
          gold: '#ffd700',
          silver: '#c0c0c0',
          bronze: '#cd7f32',
          ethereum: '#627eea',
          bitcoin: '#f7931a',
        },
        
        // RLHF Training colors
        training: {
          positive: '#10b981',
          negative: '#ef4444',
          neutral: '#6b7280',
          pending: '#f59e0b',
          complete: '#059669',
        },
        
        // Extended semantic colors
        background: {
          primary: '#0f0f23',
          secondary: '#1a1a3e',
          tertiary: '#2d1b69',
          glass: 'rgba(15, 15, 35, 0.9)',
          card: 'rgba(15, 15, 35, 0.6)',
          modal: 'rgba(26, 26, 62, 0.9)',
        },
        
        // Text colors
        text: {
          primary: '#e2e8f0',
          secondary: '#94a3b8',
          tertiary: '#64748b',
          muted: '#475569',
          accent: '#6366f1',
        },
        
        // Border colors
        border: {
          primary: 'rgba(99, 102, 241, 0.2)',
          secondary: 'rgba(99, 102, 241, 0.3)',
          accent: 'rgba(16, 185, 129, 0.3)',
          warning: 'rgba(245, 158, 11, 0.3)',
        },
      },
      
      // Custom gradients
      backgroundImage: {
        'oni-gradient': 'linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #2d1b69 100%)',
        'oni-card': 'linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1))',
        'oni-button': 'linear-gradient(135deg, #6366f1, #8b5cf6)',
        'oni-success': 'linear-gradient(135deg, #10b981, #059669)',
        'oni-warning': 'linear-gradient(135deg, #f59e0b, #d97706)',
        'oni-glass': 'linear-gradient(135deg, rgba(26, 26, 62, 0.8) 0%, rgba(15, 15, 35, 0.9) 100%)',
        'earnings-bg': 'linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05))',
        'training-bg': 'linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(16, 185, 129, 0.1))',
      },
      
      // Typography
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Monaco', 'monospace'],
        display: ['Inter', 'system-ui', 'sans-serif'],
      },
      
      // Font sizes
      fontSize: {
        'xs': ['0.75rem', { lineHeight: '1rem' }],
        'sm': ['0.875rem', { lineHeight: '1.25rem' }],
        'base': ['1rem', { lineHeight: '1.5rem' }],
        'lg': ['1.125rem', { lineHeight: '1.75rem' }],
        'xl': ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
        '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
        '5xl': ['3rem', { lineHeight: '1' }],
      },
      
      // Spacing
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      
      // Border radius
      borderRadius: {
        'xl': '1rem',
        '2xl': '1.5rem',
        '3xl': '2rem',
      },
      
      // Box shadows
      boxShadow: {
        'oni': '0 4px 12px rgba(99, 102, 241, 0.4)',
        'oni-lg': '0 8px 25px rgba(99, 102, 241, 0.4)',
        'success': '0 4px 12px rgba(16, 185, 129, 0.4)',
        'glass': '0 8px 32px rgba(0, 0, 0, 0.12)',
        'glow': '0 0 20px rgba(99, 102, 241, 0.3)',
        'inner-glow': 'inset 0 2px 4px rgba(99, 102, 241, 0.1)',
      },
      
      // Backdrop blur
      backdropBlur: {
        'xs': '2px',
        'sm': '4px',
        'md': '8px',
        'lg': '12px',
        'xl': '16px',
        '2xl': '24px',
        '3xl': '40px',
      },
      
      // Animations
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 2s infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'slide-in': 'slideIn 0.3s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-in-out',
        'fade-in': 'fadeIn 0.3s ease-in-out',
        'scale-in': 'scaleIn 0.2s ease-in-out',
        'token-reward': 'tokenReward 2s ease-in-out',
        'typing': 'typing 1.5s steps(3, end) infinite',
      },
      
      // Keyframes
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(99, 102, 241, 0.3)' },
          '100%': { boxShadow: '0 0 20px rgba(99, 102, 241, 0.8)' },
        },
        slideIn: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(40px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.9)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        tokenReward: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '50%': { opacity: '1', transform: 'translateY(0px)' },
          '100%': { opacity: '0', transform: 'translateY(-10px)' },
        },
        typing: {
          '0%': { content: '' },
          '33%': { content: '.' },
          '66%': { content: '..' },
          '100%': { content: '...' },
        },
      },
      
      // Custom utilities
      transitionTimingFunction: {
        'bounce-in': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
        'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
      },
      
      // Screen sizes for responsive design
      screens: {
        'xs': '475px',
        'sm': '640px',
        'md': '768px',
        'lg': '1024px',
        'xl': '1280px',
        '2xl': '1536px',
        '3xl': '1920px',
      },
      
      // Z-index scale
      zIndex: {
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      },
    },
  },
  plugins: [
    // Custom scrollbar plugin
    function({ addUtilities }) {
      const newUtilities = {
        '.scrollbar-hide': {
          'scrollbar-width': 'none',
          '-ms-overflow-style': 'none',
          '&::-webkit-scrollbar': {
            display: 'none',
          },
        },
        '.scrollbar-thin': {
          'scrollbar-width': 'thin',
          '&::-webkit-scrollbar': {
            width: '6px',
          },
          '&::-webkit-scrollbar-track': {
            background: 'rgba(15, 15, 35, 0.3)',
            borderRadius: '3px',
          },
          '&::-webkit-scrollbar-thumb': {
            background: 'rgba(99, 102, 241, 0.3)',
            borderRadius: '3px',
            '&:hover': {
              background: 'rgba(99, 102, 241, 0.5)',
            },
          },
        },
        '.glass-effect': {
          background: 'rgba(15, 15, 35, 0.9)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(99, 102, 241, 0.2)',
        },
        '.glow-effect': {
          boxShadow: '0 0 20px rgba(99, 102, 241, 0.3)',
          animation: 'glow 2s ease-in-out infinite alternate',
        },
      };
      addUtilities(newUtilities);
    },
  ],
  // Dark mode configuration
  darkMode: 'class',
};
