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
