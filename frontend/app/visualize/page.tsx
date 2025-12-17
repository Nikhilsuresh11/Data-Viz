'use client';
import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { api } from '@/lib/api';

// Sticky loading for Plotly
const Plot = dynamic(() => import('react-plotly.js'), {
    ssr: false,
    loading: () => <div className="h-96 w-full flex items-center justify-center bg-gray-50 rounded-xl"><div className="animate-spin h-8 w-8 border-b-2 border-primary-600 rounded-full"></div></div>
});

export default function VisualizePage() {
    const [session, setSession] = useState<any>(null);
    const [recommendations, setRecommendations] = useState<any[]>([]);
    const [chartData, setChartData] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [recLoading, setRecLoading] = useState(true);
    const [error, setError] = useState<string>('');
    const [activeChart, setActiveChart] = useState<string>('');

    useEffect(() => {
        checkSession();
    }, []);

    const checkSession = async () => {
        try {
            const data = await api.getSession();
            setSession(data);
            if (data.has_data) {
                loadRecommendations();
            }
        } catch (err) {
            console.error('Session check failed:', err);
        }
    };

    const loadRecommendations = async () => {
        try {
            const data = await api.getRecommendations();
            setRecommendations(data.recommendations || []);
            // Auto-load first recommendation
            if (data.recommendations && data.recommendations.length > 0) {
                createChart(data.recommendations[0].type, data.recommendations[0]);
            }
        } catch (err) {
            console.error('Failed to load recommendations:', err);
        } finally {
            setRecLoading(false);
        }
    };

    const createChart = async (type: string, config: any) => {
        setLoading(true);
        setError('');
        setActiveChart(config.title);

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
            <div className="container mx-auto px-4 py-20 max-w-2xl text-center">
                <div className="glass-card p-12">
                    <h1 className="text-3xl font-bold mb-4 text-gray-800">No Data Loaded</h1>
                    <a href="/upload" className="px-8 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-semibold shadow-md transition-all">Go to Upload</a>
                </div>
            </div>
        );
    }

    return (
        <div className="container mx-auto px-4 py-8">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-3xl font-bold text-gray-800">Visualizations</h1>
                <div className="text-sm text-gray-500">
                    {recommendations.length} AI Suggestions
                </div>
            </div>

            <div className="grid lg:grid-cols-4 gap-6">
                {/* Sidebar / Recommendations List */}
                <div className="lg:col-span-1 space-y-4 order-2 lg:order-1">
                    <h3 className="font-semibold text-gray-700 mb-2 px-1">Suggested Charts</h3>
                    {recLoading ? (
                        <div className="space-y-3">
                            {[1, 2, 3].map(i => <div key={i} className="h-20 bg-gray-100 rounded-lg animate-pulse"></div>)}
                        </div>
                    ) : (
                        <div className="space-y-3 max-h-[600px] overflow-y-auto px-1 pb-2 custom-scrollbar">
                            {recommendations.map((rec, i) => (
                                <div
                                    key={i}
                                    onClick={() => createChart(rec.type, rec)}
                                    className={`p-4 rounded-xl cursor-pointer transition-all border ${activeChart === rec.title
                                            ? 'bg-primary-50 border-primary-300 shadow-sm'
                                            : 'bg-white border-transparent hover:bg-gray-50 hover:shadow-sm'
                                        }`}
                                >
                                    <h4 className={`font-semibold text-sm mb-1 ${activeChart === rec.title ? 'text-primary-700' : 'text-gray-800'}`}>
                                        {rec.title}
                                    </h4>
                                    <p className="text-xs text-gray-500 line-clamp-2">{rec.description}</p>
                                    <div className="mt-2 flex gap-1">
                                        <span className="text-[10px] px-2 py-0.5 bg-gray-100 text-gray-500 rounded-full font-medium uppercase">
                                            {rec.type}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Main Chart Area */}
                <div className="lg:col-span-3 order-1 lg:order-2">
                    <div className="glass-card p-6 min-h-[500px] flex flex-col">
                        {error && (
                            <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-4 text-sm">
                                {error}
                            </div>
                        )}

                        {loading ? (
                            <div className="flex-grow flex items-center justify-center">
                                <div className="text-center">
                                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mb-4"></div>
                                    <p className="text-gray-500">Generating visualization...</p>
                                </div>
                            </div>
                        ) : chartData ? (
                            <div className="w-full h-full flex-grow">
                                <h2 className="text-xl font-bold text-gray-800 mb-4 text-center">{activeChart}</h2>
                                <Plot
                                    data={chartData.data}
                                    layout={{
                                        ...chartData.layout,
                                        autosize: true,
                                        margin: { l: 50, r: 20, t: 40, b: 50 },
                                        paper_bgcolor: 'rgba(0,0,0,0)',
                                        plot_bgcolor: 'rgba(0,0,0,0)',
                                        font: { family: 'Segoe UI, sans-serif' }
                                    }}
                                    useResizeHandler
                                    style={{ width: '100%', height: '100%', minHeight: '500px' }}
                                    config={{ responsive: true, displayModeBar: true }}
                                />
                            </div>
                        ) : (
                            <div className="flex-grow flex items-center justify-center text-gray-400">
                                <p>Select a chart from the sidebar to visualize</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
