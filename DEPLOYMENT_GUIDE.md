# Data-Viz Deployment Guide

## 1. Backend Deployment (Render)

The backend is optimized for Render's **Free Tier** (512MB RAM).

1. **Create Checkpoint**:
   Push your code to GitHub/GitLab.

2. **Create New Web Service on Render**:
   - Connect your repository.
   - **Root Directory**: `.` (Project Root)
   - **Build Command**: `bash build.sh`
   - **Start Command**: `cd backend && gunicorn app:app`
   - **Environment**: Python 3
   - **Plan**: Free

3. **Environment Variables** (Add these in Render Dashboard):
   - `PYTHON_VERSION`: `3.11.5`
   - `TOGETHER_API_KEY`: `your_key_here`
   - `SECRET_KEY`: `generate_secure_random_string`
   - `MAX_ROWS_LIMIT`: `100000`
   - `CHUNK_SIZE`: `10000`
   - `ENABLE_MEMORY_MONITORING`: `true`

## 2. Frontend Deployment (Vercel/Netlify)

The frontend is a static Next.js site.

1. **Vercel (Recommended)**:
   - Import your repository.
   - **Root Directory**: `frontend`
   - **Framework Preset**: Next.js
   - **Build Command**: `next build`
   - **Output Directory**: `.next` (default)
   - **Environment Variables**:
     - `NEXT_PUBLIC_API_URL`: `https://your-render-backend-url.onrender.com`

2. **Netlify**:
   - **Base directory**: `frontend`
   - **Build command**: `npm run build`
   - **Publish directory**: `.next`
   - **Environment Variables**:
     - `NEXT_PUBLIC_API_URL`: `https://your-render-backend-url.onrender.com`

---

## Local Development

### 1. Backend
```bash
cd backend
# Windows
..\venv\Scripts\python.exe app.py
# Server running at http://localhost:5000
```

### 2. Frontend
```bash
cd frontend
npm run dev
# Server running at http://localhost:3000
```
