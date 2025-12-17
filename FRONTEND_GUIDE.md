# Data-Viz Frontend - Minimal Next.js Implementation Guide

## Quick Setup

```bash
cd frontend
npx create-next-app@latest . --typescript --tailwind --app --no-src-dir
npm install react-plotly.js plotly.js
```

## Project Structure

```
frontend/
├── app/
│   ├── page.tsx              # Landing page
│   ├── layout.tsx            # Root layout
│   ├── upload/
│   │   └── page.tsx          # File upload
│   ├── analyze/
│   │   └── page.tsx          # Data analysis
│   └── visualize/
│       └── page.tsx          # Visualizations
├── lib/
│   └── api.ts                # API client
├── components/
│   ├── FileUploader.tsx      # Upload component
│   ├── ChartDisplay.tsx      # Chart component
│   └── LoadingSpinner.tsx    # Loading state
├── package.json
└── next.config.js
```

## 1. API Client (`lib/api.ts`)

```typescript
// lib/api.ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export const api = {
  // Upload file
  uploadFile: async (file: File, options = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    Object.entries(options).forEach(([key, value]) => {
      formData.append(key, String(value));
    });
    
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData,
      credentials: 'include'
    });
    
    if (!res.ok) throw new Error('Upload failed');
    return res.json();
  },
  
  // Analyze data
  analyzeData: async () => {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!res.ok) throw new Error('Analysis failed');
    return res.json();
  },
  
  // Get visualization recommendations
  getRecommendations: async () => {
    const res = await fetch(`${API_BASE}/api/recommendations`, {
      credentials: 'include'
    });
    
    if (!res.ok) throw new Error('Failed to get recommendations');
    return res.json();
  },
  
  // Create visualization
  createVisualization: async (type: string, config: any) => {
    const res = await fetch(`${API_BASE}/api/visualize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type, config }),
      credentials: 'include'
    });
    
    if (!res.ok) throw new Error('Visualization failed');
    return res.json();
  },
  
  // Create custom chart
  createCustomChart: async (chartType: string, config: any) => {
    const res = await fetch(`${API_BASE}/api/chart/custom`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chart_type: chartType, config }),
      credentials: 'include'
    });
    
    if (!res.ok) throw new Error('Chart creation failed');
    return res.json();
  },
  
  // Get insights
  getInsights: async () => {
    const res = await fetch(`${API_BASE}/api/insights`, {
      credentials: 'include'
    });
    
    if (!res.ok) throw new Error('Failed to get insights');
    return res.json();
  },
  
  // Chat with LLM
  chat: async (question: string) => {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      credentials: 'include'
    });
    
    if (!res.ok) throw new Error('Chat failed');
    return res.json();
  },
  
  // Get session info
  getSession: async () => {
    const res = await fetch(`${API_BASE}/api/session`, {
      credentials: 'include'
    });
    
    if (!res.ok) throw new Error('Failed to get session');
    return res.json();
  },
  
  // Clear session
  clearSession: async () => {
    const res = await fetch(`${API_BASE}/api/session/clear`, {
      method: 'POST',
      credentials: 'include'
    });
    
    if (!res.ok) throw new Error('Failed to clear session');
    return res.json();
  }
};
```

## 2. Landing Page (`app/page.tsx`)

```typescript
// app/page.tsx
import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Data-Viz
          </h1>
          <p className="text-xl text-gray-600">
            AI-Powered Data Analysis & Visualization
          </p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
          <Link href="/upload" className="bg-white p-8 rounded-lg shadow-lg hover:shadow-xl transition-shadow">
            <div className="text-4xl mb-4">📊</div>
            <h2 className="text-2xl font-semibold mb-2">Upload</h2>
            <p className="text-gray-600">Upload your CSV or Excel file to get started</p>
          </Link>
          
          <Link href="/analyze" className="bg-white p-8 rounded-lg shadow-lg hover:shadow-xl transition-shadow">
            <div className="text-4xl mb-4">🔍</div>
            <h2 className="text-2xl font-semibold mb-2">Analyze</h2>
            <p className="text-gray-600">Get AI-powered insights about your data</p>
          </Link>
          
          <Link href="/visualize" className="bg-white p-8 rounded-lg shadow-lg hover:shadow-xl transition-shadow">
            <div className="text-4xl mb-4">📈</div>
            <h2 className="text-2xl font-semibold mb-2">Visualize</h2>
            <p className="text-gray-600">Create beautiful interactive charts</p>
          </Link>
        </div>
        
        <div className="mt-16 text-center text-sm text-gray-500">
          <p>Optimized for Render Free Tier • Lazy Loading • Memory Efficient</p>
        </div>
      </div>
    </div>
  );
}
```

## 3. Upload Page (`app/upload/page.tsx`)

```typescript
// app/upload/page.tsx
'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string>('');
  
  const [options, setOptions] = useState({
    handleMissingValues: true,
    convertDateColumns: true,
    handleOutliers: true
  });

  const handleUpload = async () => {
    if (!file) return;
    
    setLoading(true);
    setError('');
    
    try {
      const data = await api.uploadFile(file, options);
      setResult(data);
      
      // Auto-navigate to analyze page after successful upload
      setTimeout(() => {
        router.push('/analyze');
      }, 2000);
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-4 max-w-2xl">
        <h1 className="text-3xl font-bold mb-8">Upload Your Data</h1>
        
        <div className="bg-white rounded-lg shadow p-6">
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select File (CSV or Excel)
            </label>
            <input
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={(e) => setFile(e.files?.[0] || null)}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-50 file:text-blue-700
                hover:file:bg-blue-100"
            />
          </div>
          
          <div className="mb-6 space-y-2">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={options.handleMissingValues}
                onChange={(e) => setOptions({...options, handleMissingValues: e.target.checked})}
                className="mr-2"
              />
              <span className="text-sm">Handle missing values</span>
            </label>
            
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={options.convertDateColumns}
                onChange={(e) => setOptions({...options, convertDateColumns: e.target.checked})}
                className="mr-2"
              />
              <span className="text-sm">Convert date columns</span>
            </label>
            
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={options.handleOutliers}
                onChange={(e) => setOptions({...options, handleOutliers: e.target.checked})}
                className="mr-2"
              />
              <span className="text-sm">Handle outliers</span>
            </label>
          </div>
          
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg
              hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
              transition-colors font-medium"
          >
            {loading ? 'Uploading...' : 'Upload & Process'}
          </button>
          
          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
              {error}
            </div>
          )}
          
          {result && (
            <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
              <h3 className="font-semibold text-green-900 mb-2">Upload Successful!</h3>
              <p className="text-sm text-green-700">Rows: {result.rows}</p>
              <p className="text-sm text-green-700">Columns: {result.columns}</p>
              <p className="text-sm text-green-500 mt-2">Redirecting to analysis...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

## 4. Analyze Page (`app/analyze/page.tsx`)

```typescript
// app/analyze/page.tsx
'use client';
import { useState, useEffect } from 'react';
import { api } from '@/lib/api';

export default function AnalyzePage() {
  const [session, setSession] = useState<any>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [insights, setInsights] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    checkSession();
  }, []);

  const checkSession = async () => {
    try {
      const data = await api.getSession();
      setSession(data);
    } catch (err) {
      console.error('Session check failed:', err);
    }
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError('');
    
    try {
      const data = await api.analyzeData();
      setAnalysis(data);
    } catch (err: any) {
      setError(err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleGetInsights = async () => {
    setLoading(true);
    setError('');
    
    try {
      const data = await api.getInsights();
      setInsights(data);
    } catch (err: any) {
      setError(err.message || 'Failed to get insights');
    } finally {
      setLoading(false);
    }
  };

  if (!session?.has_data) {
    return (
      <div className="min-h-screen bg-gray-50 py-12">
        <div className="container mx-auto px-4 max-w-2xl text-center">
          <h1 className="text-3xl font-bold mb-4">No Data Loaded</h1>
          <p className="text-gray-600 mb-6">Please upload a file first</p>
          <a href="/upload" className="bg-blue-600 text-white py-2 px-6 rounded-lg hover:bg-blue-700">
            Go to Upload
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-4 max-w-4xl">
        <h1 className="text-3xl font-bold mb-8">Analyze Data</h1>
        
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Dataset Info</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Rows</p>
              <p className="text-2xl font-bold">{session.rows?.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Columns</p>
              <p className="text-2xl font-bold">{session.columns}</p>
            </div>
          </div>
        </div>
        
        <div className="flex gap-4 mb-6">
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="bg-green-600 text-white py-3 px-6 rounded-lg
              hover:bg-green-700 disabled:bg-gray-300 transition-colors"
          >
            {loading ? 'Analyzing...' : 'Run Analysis'}
          </button>
          
          <button
            onClick={handleGetInsights}
            disabled={loading}
            className="bg-purple-600 text-white py-3 px-6 rounded-lg
              hover:bg-purple-700 disabled:bg-gray-300 transition-colors"
          >
            {loading ? 'Loading...' : 'Get AI Insights'}
          </button>
        </div>
        
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}
        
        {analysis && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>
            <pre className="bg-gray-50 p-4 rounded overflow-auto max-h-96 text-sm">
              {JSON.stringify(analysis, null, 2)}
            </pre>
          </div>
        )}
        
        {insights && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">AI Insights</h2>
            <ul className="space-y-2">
              {insights.insights?.map((insight: string, i: number) => (
                <li key={i} className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span>{insight}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
```

## 5. Visualize Page (`app/visualize/page.tsx`)

```typescript
// app/visualize/page.tsx
'use client';
import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { api } from '@/lib/api';

// Lazy load Plotly to reduce bundle size
const Plot = dynamic(() => import('react-plotly.js'), { 
  ssr: false,
  loading: () => <div className="text-center py-8">Loading chart...</div>
});

export default function VisualizePage() {
  const [session, setSession] = useState<any>(null);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [chartData, setChartData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    checkSession();
    loadRecommendations();
  }, []);

  const checkSession = async () => {
    try {
      const data = await api.getSession();
      setSession(data);
    } catch (err) {
      console.error('Session check failed:', err);
    }
  };

  const loadRecommendations = async () => {
    try {
      const data = await api.getRecommendations();
      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error('Failed to load recommendations:', err);
    }
  };

  const createChart = async (type: string, config: any) => {
    setLoading(true);
    setError('');
    
    try {
      const data = await api.createVisualization(type, config);
      if (data.success) {
        setChartData(data.data);
      } else {
        setError(data.error || 'Failed to create chart');
      }
    } catch (err: any) {
      setError(err.message || 'Chart creation failed');
    } finally {
      setLoading(false);
    }
  };

  if (!session?.has_data) {
    return (
      <div className="min-h-screen bg-gray-50 py-12">
        <div className="container mx-auto px-4 max-w-2xl text-center">
          <h1 className="text-3xl font-bold mb-4">No Data Loaded</h1>
          <p className="text-gray-600 mb-6">Please upload a file first</p>
          <a href="/upload" className="bg-blue-600 text-white py-2 px-6 rounded-lg hover:bg-blue-700">
            Go to Upload
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-4 max-w-6xl">
        <h1 className="text-3xl font-bold mb-8">Visualize Data</h1>
        
        {recommendations.length > 0 && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Recommended Visualizations</h2>
            <div className="grid md:grid-cols-2 gap-4">
              {recommendations.slice(0, 6).map((rec, i) => (
                <button
                  key={i}
                  onClick={() => createChart(rec.type, rec)}
                  disabled={loading}
                  className="text-left p-4 border rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors disabled:opacity-50"
                >
                  <h3 className="font-semibold text-blue-600">{rec.title}</h3>
                  <p className="text-sm text-gray-600 mt-1">{rec.description}</p>
                </button>
              ))}
            </div>
          </div>
        )}
        
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}
        
        {loading && (
          <div className="text-center py-8">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            <p className="mt-4 text-gray-600">Creating visualization...</p>
          </div>
        )}
        
        {chartData && !loading && (
          <div className="bg-white rounded-lg shadow p-6">
            <Plot
              data={chartData.data}
              layout={{
                ...chartData.layout,
                autosize: true,
                responsive: true
              }}
              useResizeHandler
              style={{ width: '100%', height: '600px' }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
```

## 6. Environment Variables

Create `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

For production:
```
NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
```

## 7. Next.js Config (`next.config.js`)

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Optimize for production
  swcMinify: true,
  // Reduce bundle size
  experimental: {
    optimizePackageImports: ['react-plotly.js', 'plotly.js']
  }
}

module.exports = nextConfig
```

## Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend will run on `http://localhost:3000`

## Key Features

✅ **Lazy Loading**: Plotly loaded only when needed  
✅ **No Prefetching**: Data fetched only on user action  
✅ **Loading States**: Clear feedback during operations  
✅ **Error Handling**: Graceful error display  
✅ **Session Management**: Checks for uploaded data  
✅ **Responsive Design**: Works on all screen sizes  

## Production Build

```bash
npm run build
npm start
```

This creates an optimized production build with minimal JavaScript bundle size.
