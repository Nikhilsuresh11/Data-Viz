'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('csv'); // 'csv', 'doc', 'url'
  const [error, setError] = useState<string>('');

  const [options, setOptions] = useState({
    handleMissingValues: true,
    convertDateColumns: true,
    handleOutliers: true,
    encodeCategorical: false
  });

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError('');

    try {
      const data = await api.uploadFile(file, options);
      router.push('/overview');
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-12 max-w-4xl">
      <div className="glass-card p-8 animate-fade-in">
        <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">Upload Your Data</h1>

        {/* Tabs */}
        <div className="flex border-b border-gray-200 mb-8">
          <button
            onClick={() => setActiveTab('csv')}
            className={`px-6 py-3 font-medium transition-colors border-b-2 flex items-center space-x-2 ${activeTab === 'csv'
              ? 'border-primary-500 text-primary-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" /><polyline points="14 2 14 8 20 8" /><path d="M8 13h2" /><path d="M8 17h2" /><path d="M14 13h2" /><path d="M14 17h2" /></svg>
            <span>CSV / Excel</span>
          </button>
          <button
            onClick={() => setActiveTab('doc')}
            className={`px-6 py-3 font-medium transition-colors border-b-2 flex items-center space-x-2 ${activeTab === 'doc'
              ? 'border-primary-500 text-primary-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" /><polyline points="14 2 14 8 20 8" /><path d="M12 18v-6" /><path d="M8 15h8" /></svg>
            <span>Document</span>
          </button>
          <button
            onClick={() => setActiveTab('url')}
            className={`px-6 py-3 font-medium transition-colors border-b-2 flex items-center space-x-2 ${activeTab === 'url'
              ? 'border-primary-500 text-primary-600'
              : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><line x1="2" y1="12" x2="22" y2="12" /><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" /></svg>
            <span>Website</span>
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'csv' && (
          <div className="space-y-8">
            <div
              className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${file ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
                }`}
            >
              <input
                type="file"
                id="file-upload"
                accept=".csv,.xlsx,.xls"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="hidden"
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                {file ? (
                  <div className="animate-fade-in">
                    <div className="mx-auto w-16 h-16 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center mb-4">
                      <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" /><polyline points="14 2 14 8 20 8" /></svg>
                    </div>
                    <h3 className="text-xl font-semibold text-gray-800">{file.name}</h3>
                    <p className="text-gray-500 mt-2">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                ) : (
                  <div>
                    <div className="mx-auto w-16 h-16 bg-gray-100 text-gray-400 rounded-full flex items-center justify-center mb-4">
                      <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
                    </div>
                    <h3 className="text-xl font-semibold text-gray-800">Drag & Drop your file here</h3>
                    <p className="text-gray-500 mt-2">or click to browse (CSV, Excel)</p>
                  </div>
                )}
              </label>
            </div>

            {/* Options */}
            <div className="bg-gray-50 rounded-lg p-6">
              <h4 className="font-semibold text-gray-700 mb-4">Processing Options</h4>
              <div className="grid md:grid-cols-2 gap-4">
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input type="checkbox" checked={options.handleMissingValues} onChange={e => setOptions({ ...options, handleMissingValues: e.target.checked })} className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500" />
                  <span className="text-gray-700">Handle missing values</span>
                </label>
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input type="checkbox" checked={options.convertDateColumns} onChange={e => setOptions({ ...options, convertDateColumns: e.target.checked })} className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500" />
                  <span className="text-gray-700">Auto-convert dates</span>
                </label>
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input type="checkbox" checked={options.handleOutliers} onChange={e => setOptions({ ...options, handleOutliers: e.target.checked })} className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500" />
                  <span className="text-gray-700">Handle outliers (IQR)</span>
                </label>
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input type="checkbox" checked={options.encodeCategorical} onChange={e => setOptions({ ...options, encodeCategorical: e.target.checked })} className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500" />
                  <span className="text-gray-700">Encode categorical data</span>
                </label>
              </div>
            </div>

            {error && (
              <div className="bg-red-50 text-red-700 p-4 rounded-lg flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg>
                {error}
              </div>
            )}

            <button
              onClick={handleUpload}
              disabled={!file || loading}
              className="w-full py-4 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-bold text-lg shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:-translate-y-0.5"
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                  Processing Data...
                </span>
              ) : 'Upload & Analyze'}
            </button>
          </div>
        )}

        {/* Placeholder text for other tabs */}
        {(activeTab === 'doc' || activeTab === 'url') && (
          <div className="text-center py-12 text-gray-500">
            <p>This feature is coming soon to the Next.js version.</p>
            <p className="text-sm mt-2">Use the CSV upload for now.</p>
          </div>
        )}
      </div>
    </div>
  );
}
