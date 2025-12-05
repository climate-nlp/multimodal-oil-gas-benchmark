# Deployment Guide for GitHub Pages

This guide explains how to deploy the dashboard to GitHub Pages from the `dashboard/` subdirectory.

## Prerequisites

- GitHub repository for your project
- Node.js and npm installed locally
- Data file: `dashboard/public/data/yt_video.all.jsonl` (make sure this exists)

## Option 1: Automated Deployment (Recommended)

### Initial Setup

1. **Enable GitHub Pages in your repository:**
   - Go to your repository on GitHub
   - Navigate to **Settings** → **Pages**
   - Under "Build and deployment", set **Source** to **GitHub Actions**

2. **Push your code:**
   ```bash
   git add .
   git commit -m "Add GitHub Pages deployment"
   git push origin main
   ```

3. **The workflow will automatically:**
   - Build your application
   - Deploy to GitHub Pages
   - Your site will be available at: `https://username.github.io/repo-name/`

### Subsequent Deployments

Simply push to the `main` branch, and the site will automatically rebuild and redeploy:
```bash
git push origin main
```

You can also manually trigger deployment from the **Actions** tab in your GitHub repository.

---

## Option 2: Manual Deployment

If you prefer to deploy manually or need to deploy from a local build:

### 1. Build the Application

```bash
# Navigate to dashboard directory
cd dashboard

# Install dependencies (if not already done)
npm install

# Build with your repository name
REPO_NAME=your-repo-name npm run build
```

Replace `your-repo-name` with your actual GitHub repository name.

### 2. Deploy using gh-pages

First, install the `gh-pages` package:
```bash
npm install --save-dev gh-pages
```

Add this script to your `dashboard/package.json`:
```json
"scripts": {
  "deploy": "REPO_NAME=your-repo-name npm run build && gh-pages -d dist"
}
```

Then deploy:
```bash
npm run deploy
```

---

## Configuration Notes

### Repository Name
- **Project site** (`username.github.io/repo-name`): Set `REPO_NAME` to your repository name
- **User/Org site** (`username.github.io`): Leave `REPO_NAME` empty or unset

### Data File Location
The application expects the data file at `dashboard/public/data/yt_video.all.jsonl`. Make sure:
1. The file exists at this location
2. It's not in `.gitignore` (or use Git LFS for large files)
3. The file is committed to your repository

---

## Troubleshooting

### Assets not loading (404 errors)
- **Problem**: CSS, JS, or data files return 404
- **Solution**: Ensure `REPO_NAME` is set correctly during build. The base path must match your repository name.

### Data file not found
- **Problem**: "Error Loading Dataset" message appears
- **Solution**: 
  1. Verify `dashboard/public/data/yt_video.all.jsonl` exists
  2. Check that the file is committed to git with `git add -f dashboard/public/data/yt_video.all.jsonl`
  3. For large files (>100MB), use Git LFS

### Workflow fails
- **Problem**: GitHub Actions workflow fails
- **Solution**:
  1. Check the Actions tab for error details
  2. Ensure GitHub Pages is enabled in repository settings
  3. Verify the workflow has proper permissions

### Site shows old version
- **Problem**: Changes don't appear after deployment
- **Solution**:
  1. Hard refresh your browser (Ctrl+Shift+R or Cmd+Shift+R)
  2. Check that the workflow completed successfully
  3. Wait a few minutes for GitHub's CDN to update

---

## Local Preview

To preview the production build locally:

```bash
# Navigate to dashboard directory
cd dashboard

# Build the app
REPO_NAME=your-repo-name npm run build

# Preview the build
npm run preview
```

This will serve the built files at `http://localhost:4173` (or similar).

---

## Need Help?

- Check the [GitHub Pages documentation](https://docs.github.com/en/pages)
- Review workflow runs in the **Actions** tab
- Ensure all dependencies are installed with `npm install`
