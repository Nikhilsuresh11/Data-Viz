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
                onChange={(e) => setOptions({ ...options, handleMissingValues: e.target.checked })}
                className="mr-2"
              />
              <span className="text-sm">Handle missing values</span>
            </label>

            <label className="flex items-center">
              <input
                type="checkbox"
                checked={options.convertDateColumns}
                onChange={(e) => setOptions({ ...options, convertDateColumns: e.target.checked })}
                className="mr-2"
              />
              <span className="text-sm">Convert date columns</span>
            </label>

            <label className="flex items-center">
              <input
                type="checkbox"
                checked={options.handleOutliers}
                onChange={(e) => setOptions({ ...options, handleOutliers: e.target.checked })}
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
