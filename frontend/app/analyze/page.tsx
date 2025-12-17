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
            if (data.has_data) {
                handleAnalyze();
            }
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
            // Fetch insights in parallel or after
            handleGetInsights();
        } catch (err: any) {
            setError(err.message || 'Analysis failed');
        } finally {
            setLoading(false);
        }
    };

    const handleGetInsights = async () => {
        try {
            const data = await api.getInsights();
            setInsights(data);
        } catch (err: any) {
            console.error('Failed to get insights', err);
        }
    };

    if (!session?.has_data) {
        return (
            <div className="container mx-auto px-4 py-20 max-w-2xl text-center">
                <div className="glass-card p-12">
                    <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6 text-gray-400">
                        <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" /><polyline points="14 2 14 8 20 8" /><line x1="12" y1="18" x2="12" y2="12" /><line x1="9" y1="15" x2="15" y2="15" /></svg>
                    </div>
                    <h1 className="text-3xl font-bold mb-4 text-gray-800">No Data Loaded</h1>
                    <p className="text-gray-600 mb-8">Please upload a file to start analyzing.</p>
                    <a href="/upload" className="px-8 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-semibold shadow-md transition-all">
                        Go to Upload
                    </a>
                </div>
            </div>
        );
    }

    return (
        <div className="container mx-auto px-4 py-8">
            <div className="flex justify-between items-center mb-8">
                <div>
                    <h1 className="text-3xl font-bold text-gray-800">Dataset Overview</h1>
                    <p className="text-gray-500 mt-1">{session?.filename || 'Uploaded Data'}</p>
                </div>
                <div className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                    Processed {new Date().toLocaleDateString()}
                </div>
            </div>

            {loading && !analysis ? (
                <div className="text-center py-20">
                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mb-4"></div>
                    <p className="text-gray-600">Analyzing your data...</p>
                </div>
            ) : (
                <div className="space-y-6">
                    {/* Metrics Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                        <div className="glass-card p-6 text-center hover:-translate-y-1 transition-transform">
                            <div className="text-3xl font-bold text-primary-600 mb-2">{session?.rows?.toLocaleString()}</div>
                            <div className="text-sm text-gray-500 uppercase tracking-wide font-semibold">Rows</div>
                        </div>
                        <div className="glass-card p-6 text-center hover:-translate-y-1 transition-transform">
                            <div className="text-3xl font-bold text-primary-600 mb-2">{session?.columns}</div>
                            <div className="text-sm text-gray-500 uppercase tracking-wide font-semibold">Columns</div>
                        </div>
                        <div className="glass-card p-6 text-center hover:-translate-y-1 transition-transform">
                            <div className="text-3xl font-bold text-primary-600 mb-2">
                                {analysis?.numeric_columns?.length || 0}
                            </div>
                            <div className="text-sm text-gray-500 uppercase tracking-wide font-semibold">Numeric Vars</div>
                        </div>
                        <div className="glass-card p-6 text-center hover:-translate-y-1 transition-transform">
                            <div className="text-3xl font-bold text-primary-600 mb-2">
                                {analysis?.categorical_columns?.length || 0}
                            </div>
                            <div className="text-sm text-gray-500 uppercase tracking-wide font-semibold">Categorical</div>
                        </div>
                    </div>

                    <div className="grid md:grid-cols-3 gap-6">
                        {/* Dataset Info */}
                        <div className="md:col-span-2 glass-card p-6">
                            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                                <svg className="w-5 h-5 mr-2 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
                                Variable Statistics
                            </h2>
                            <div className="space-y-4 max-h-96 overflow-y-auto pr-2 custom-scrollbar">
                                {analysis?.summary && Object.entries(analysis.summary).map(([col, stats]: [string, any]) => (
                                    <div key={col} className="border-b border-gray-100 last:border-0 pb-4 last:pb-0">
                                        <div className="flex justify-between items-center mb-2">
                                            <h3 className="font-semibold text-gray-700">{col}</h3>
                                            <span className="text-xs bg-gray-100 px-2 py-0.5 rounded text-gray-500">
                                                {typeof stats.mean !== 'undefined' ? 'Numeric' : 'Categorical'}
                                            </span>
                                        </div>
                                        <div className="grid grid-cols-2 gap-4 text-sm">
                                            {stats.mean ? (
                                                <>
                                                    <div className="text-gray-600">Mean: <span className="font-medium text-gray-900">{stats.mean.toFixed(2)}</span></div>
                                                    <div className="text-gray-600">Max: <span className="font-medium text-gray-900">{stats.max.toFixed(2)}</span></div>
                                                </>
                                            ) : (
                                                <div className="text-gray-600 col-span-2">
                                                    Unique Values: <span className="font-medium text-gray-900">{stats.unique || 'N/A'}</span>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* AI Insights Panel */}
                        <div className="glass-card p-6 bg-gradient-to-br from-white to-primary-50">
                            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                                <svg className="w-5 h-5 mr-2 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                                AI Insights
                            </h2>
                            {insights ? (
                                <ul className="space-y-3">
                                    {insights.insights?.slice(0, 5).map((insight: string, i: number) => (
                                        <li key={i} className="flex items-start text-sm text-gray-700 bg-white p-3 rounded-lg shadow-sm border border-primary-100">
                                            <span className="text-primary-500 mr-2 mt-0.5">•</span>
                                            {insight}
                                        </li>
                                    ))}
                                </ul>
                            ) : (
                                <div className="text-center py-8 text-gray-400">
                                    Loading insights...
                                </div>
                            )}

                            <div className="mt-6 pt-4 border-t border-primary-100 text-center">
                                <a href="/visualize" className="inline-block w-full py-2 bg-white border border-primary-200 text-primary-600 rounded-lg hover:bg-primary-50 transition-colors font-medium">
                                    View Visualizations
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
