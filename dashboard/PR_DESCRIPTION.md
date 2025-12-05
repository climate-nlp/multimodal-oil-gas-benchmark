# Add GitHub Pages Deployment for Interactive Dashboard

## Overview

This PR adds a GitHub Pages deployment setup for the ashboard, enabling the interactive visualization to be hosted directly from this repository.

## Changes Made

### 1. Vite Configuration (`vite.config.ts`)
- Added configurable `base` path for GitHub Pages deployment
- Automatically uses repository name from environment variable
- Maintains compatibility with local development

### 2. GitHub Actions Workflow (`.github/workflows/deploy.yml`)
- Automated deployment on push to `main` branch
- Builds the application with correct base path
- Deploys to GitHub Pages automatically
- Supports manual triggering

### 3. Documentation (`DEPLOYMENT.md`)
- Comprehensive deployment guide
- Instructions for both automated and manual deployment
- Troubleshooting section

### 4. Data File Management
- Updated `.gitignore` to exclude data file from multiple locations
- Data file (`public/data/yt_video.all.jsonl`) is included in build output

## What the Repository Owner Needs to Do

### Step 1: Enable GitHub Pages (Required)

After merging this PR, the repository owner must enable GitHub Pages:

1. Go to repository **Settings** → **Pages**
2. Under "Build and deployment":
   - Set **Source** to **GitHub Actions**
3. Save the settings

That's it! The workflow will automatically deploy on the next push to `main`.

### Step 2: Verify Deployment

After enabling GitHub Pages and the workflow runs:
- The site will be available at: `https://[username].github.io/[repo-name]/`
- Check the **Actions** tab to monitor deployment progress
- First deployment may take 2-3 minutes

## Testing

✅ Production build tested locally:
```bash
REPO_NAME=climate-ad-monitor npm run build
```

Build output verified:
- Assets correctly bundled with base path
- Data file (171 KB) successfully copied to build
- No build errors or warnings (except chunk size advisory)

## Notes

- The workflow automatically uses the repository name as the base path
- Data file is currently 171 KB (well under GitHub's limits)
- If the data file grows beyond 100 MB in the future, Git LFS will be needed

## Preview

Once deployed, users will be able to:
- View interactive framing distribution charts
- Explore yearly trends across different framing categories
- Access aggregate sector analysis
- All without needing to clone or run the project locally

## Questions?

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment documentation and troubleshooting.
