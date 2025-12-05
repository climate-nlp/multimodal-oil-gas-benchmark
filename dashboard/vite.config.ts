import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// For GitHub Pages deployment, set REPO_NAME env variable to your repo name
// e.g., REPO_NAME=my-repo npm run build
// If deploying to username.github.io (user site), leave REPO_NAME empty
const base = process.env.REPO_NAME ? `/${process.env.REPO_NAME}/` : '/'

export default defineConfig({
  plugins: [react()],
  base: base,
})