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
