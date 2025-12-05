## Quick Start

### Prerequisites
- Node.js 18+ 
- npm

### Installation

```bash
cd dashboard
npm install
```

### Development

```bash
npm run dev
```

Visit `http://localhost:5173` to view the dashboard.

### Build for Production

```bash
npm run build
npm run preview
```

## Data

The dashboard loads data from `public/data/yt_video.all.jsonl`.

To update the data file:
```bash
# Copy your new data file
cp /path/to/new/yt_video.all.jsonl public/data/

# Force add to git (it's in .gitignore)
git add -f public/data/yt_video.all.jsonl
git commit -m "Update dataset"
```

## Deployment

The dashboard is automatically deployed to GitHub Pages when changes are pushed to the `main` branch. See [DEPLOYMENT.md](./DEPLOYMENT.md) for details.

**Live Demo**: `https://[username].github.io/[repo-name]/`

## Project Structure

```
dashboard/
├── src/
│   ├── components/      # React components (charts, cards)
│   ├── pages/          # Dashboard and References pages
│   ├── services/       # Data loading and processing
│   └── types.ts        # TypeScript type definitions
├── public/
│   └── data/          # Dataset (JSONL format)
├── index.html         # Entry point
├── vite.config.ts     # Build configuration
└── package.json       # Dependencies
```

## Technology Stack

- **Frontend**: React 18 + TypeScript
- **Visualization**: Recharts
- **Styling**: Tailwind CSS
- **Build**: Vite
- **Deployment**: GitHub Pages

## Citation

When referencing this dashboard or its findings, please cite:

```
@inproceedings{morio-etal-2025-multimodal,
	author = {Morio, Gaku and Rowlands, Harri and Stammbach, Dominik and Manning, Christopher D and Henderson, Peter},
	booktitle = {Advances in Neural Information Processing Systems},
	title = {A Multimodal Benchmark for Framing of Oil \& Gas Advertising and Potential Greenwashing Detection},
	year = {2025}
}
```

## License

See [LICENSE.txt](../LICENSE.txt) in the root directory.

## Related

- **Paper Reproduction Code**: See the root directory for Python scripts and data processing code
- **Full Documentation**: [DEPLOYMENT.md](./DEPLOYMENT.md) for deployment details
- **References**: See the full NeurIPS paper