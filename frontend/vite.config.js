import { defineConfig } from 'vite'
import dotenv from 'dotenv';

// Only load .env file in local development — don't throw if it's missing
dotenv.config({ path: '../.env' });

export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
