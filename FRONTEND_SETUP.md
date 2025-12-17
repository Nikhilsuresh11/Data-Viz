# Frontend Setup - Quick Start Guide

## Automatic Setup (Recommended)

Run these commands in PowerShell:

```powershell
cd frontend

# Create package.json
@"
{
  "name": "data-viz-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev -p 3000",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "14.2.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-plotly.js": "^2.6.0",
    "plotly.js": "^2.29.0"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "typescript": "^5",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.49",
    "tailwindcss": "^3.4.17"
  }
}
"@ | Out-File -FilePath package.json -Encoding UTF8

# Install dependencies
npm install

# Create tsconfig.json
@"
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{"name": "next"}],
    "paths": {"@/*": ["./*"]}
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
"@ | Out-File -FilePath tsconfig.json -Encoding UTF8

# Create next.config.js
@"
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true
}
module.exports = nextConfig
"@ | Out-File -FilePath next.config.js -Encoding UTF8

# Create tailwind.config.js
@"
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {extend: {}},
  plugins: [],
}
"@ | Out-File -FilePath tailwind.config.js -Encoding UTF8

# Create postcss.config.js
@"
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
"@ | Out-File -FilePath postcss.config.js -Encoding UTF8

# Create .env.local
@"
NEXT_PUBLIC_API_URL=http://localhost:5000
"@ | Out-File -FilePath .env.local -Encoding UTF8

# Create globals.css
New-Item -ItemType Directory -Force -Path app | Out-Null
@"
@tailwind base;
@tailwind components;
@tailwind utilities;
"@ | Out-File -FilePath app/globals.css -Encoding UTF8

# Create root layout
@"
import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Data-Viz - AI-Powered Data Analysis',
  description: 'Upload, analyze, and visualize your data with AI',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
"@ | Out-File -FilePath app/layout.tsx -Encoding UTF8

Write-Host "Frontend setup complete!"
Write-Host "Now copy the code from FRONTEND_GUIDE.md to create the pages"
```

## Manual File Creation

If automatic setup doesn't work, create these files manually:

### 1. `frontend/package.json`
See FRONTEND_GUIDE.md section "Quick Setup"

### 2. `frontend/tsconfig.json`
Standard Next.js TypeScript config

### 3. `frontend/next.config.js`
```javascript
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true
}
module.exports = nextConfig
```

### 4. `frontend/tailwind.config.js`
Standard Tailwind config for Next.js app directory

### 5. `frontend/.env.local`
```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### 6. `frontend/app/globals.css`
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

### 7. `frontend/app/layout.tsx`
Root layout with metadata

## After Setup

1. **Install dependencies**:
   ```powershell
   cd frontend
   npm install
   ```

2. **Copy code from FRONTEND_GUIDE.md**:
   - `lib/api.ts` - API client
   - `app/page.tsx` - Landing page
   - `app/upload/page.tsx` - Upload page
   - `app/analyze/page.tsx` - Analyze page
   - `app/visualize/page.tsx` - Visualize page

3. **Run development server**:
   ```powershell
   npm run dev
   ```

4. **Access frontend**: http://localhost:3000

## Testing Full Stack

1. **Backend**: http://localhost:5000 (already running)
2. **Frontend**: http://localhost:3000 (after npm run dev)
3. **Upload a CSV file** through the frontend
4. **Analyze and visualize** your data

## Production Build

```powershell
npm run build
npm start
```

## Troubleshooting

**Port already in use**:
```powershell
npm run dev -- -p 3001
```

**Module not found**:
```powershell
npm install
```

**CORS errors**:
- Backend already has CORS enabled for localhost:3000
- Check `.env.local` has correct API URL

## Quick Test

After setup, test the API connection:

```typescript
// In browser console at localhost:3000
fetch('http://localhost:5000/health')
  .then(r => r.json())
  .then(console.log)
```

Should return: `{status: 'healthy', service: 'data-viz-backend'}`
