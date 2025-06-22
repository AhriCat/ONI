import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react({
      // Enable React Fast Refresh
      fastRefresh: true,
      // Include JSX runtime
      jsxRuntime: 'automatic',
    }),
  ],
  
  // Path resolution
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@/components': path.resolve(__dirname, './src/components'),
      '@/hooks': path.resolve(__dirname, './src/hooks'),
      '@/services': path.resolve(__dirname, './src/services'),
      '@/types': path.resolve(__dirname, './src/types'),
      '@/contexts': path.resolve(__dirname, './src/contexts'),
      '@/utils': path.resolve(__dirname, './src/utils'),
      '@/styles': path.resolve(__dirname, './src/styles'),
    },
  },

  // Development server configuration
  server: {
    port: 3000,
    host: true,
    open: true,
    cors: true,
    proxy: {
      // Proxy ONI blockchain API calls
      '/api/blockchain': {
        target: 'http://localhost:8545',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/blockchain/, ''),
      },
      // Proxy RLHF training API calls
      '/api/rlhf': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/rlhf/, '/rlhf'),
      },
      // Proxy chat API calls
      '/api/chat': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/chat/, '/chat'),
      },
      // WebSocket for real-time features
      '/socket.io': {
        target: 'http://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },

  // Build configuration
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    minify: 'terser',
    target: 'es2020',
    
    // Chunk splitting for better caching
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunks
          'react-vendor': ['react', 'react-dom'],
          'web3-vendor': ['ethers', 'wagmi', 'viem', '@rainbow-me/rainbowkit'],
          'ui-vendor': ['framer-motion', 'lucide-react', 'react-hot-toast'],
          'chart-vendor': ['recharts'],
          'utils-vendor': ['date-fns', 'clsx', 'tailwind-merge'],
        },
      },
    },
    
    // Terser options for minification
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
      },
    },
    
    // Asset optimization
    assetsInlineLimit: 4096,
    chunkSizeWarningLimit: 500,
  },

  // Environment variables
  define: {
    // ONI blockchain configuration
    __ONI_CHAIN_ID__: JSON.stringify(process.env.VITE_ONI_CHAIN_ID || '1337'),
    __ONI_RPC_URL__: JSON.stringify(process.env.VITE_ONI_RPC_URL || 'http://localhost:8545'),
    __ONI_CONTRACT_ADDRESS__: JSON.stringify(process.env.VITE_ONI_CONTRACT_ADDRESS || ''),
    __RLHF_API_URL__: JSON.stringify(process.env.VITE_RLHF_API_URL || 'http://localhost:8000'),
    __CHAT_API_URL__: JSON.stringify(process.env.VITE_CHAT_API_URL || 'http://localhost:8000'),
    __WALLETCONNECT_PROJECT_ID__: JSON.stringify(process.env.VITE_WALLETCONNECT_PROJECT_ID || ''),
    
    // Feature flags
    __ENABLE_MAINNET__: JSON.stringify(process.env.VITE_ENABLE_MAINNET === 'true'),
    __ENABLE_TESTNET__: JSON.stringify(process.env.VITE_ENABLE_TESTNET === 'true'),
    __ENABLE_ANALYTICS__: JSON.stringify(process.env.VITE_ENABLE_ANALYTICS === 'true'),
  },

  // CSS configuration
  css: {
    devSourcemap: true,
    postcss: {
      plugins: [
        require('tailwindcss'),
        require('autoprefixer'),
      ],
    },
  },

  // Optimization
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'ethers',
      'wagmi',
      'framer-motion',
      'lucide-react',
      'recharts',
      'socket.io-client',
    ],
    exclude: ['@wagmi/core'],
  },

  // Worker configuration for Web3 operations
  worker: {
    format: 'es',
  },

  // Preview server configuration (for production builds)
  preview: {
    port: 3000,
    host: true,
    cors: true,
  },

  // Test configuration
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: true,
  },
});
