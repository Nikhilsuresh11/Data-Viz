'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { api } from '@/lib/api';

const Plot = dynamic(() => import('react-plotly.js'), {
    ssr: false,
    loading: () => <div className="h-full w-full flex items-center justify-center text-gray-400">Loading Chart Engine...</div>
});

const CHART_TYPES = [
    { id: 'bar', label: 'Bar Chart', icon: 'M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2zm4-16v12m4-8v8m4-10v10' },
    { id: 'line', label: 'Line Chart', icon: 'M22 12h-4l-3 9L9 3l-3 9H2' },
    { id: 'scatter', label: 'Scatter Plot', icon: 'M12 12a2 2 0 1 0 0-4 2 2 0 0 0 0 4zm-7 7a2 2 0 1 0 0-4 2 2 0 0 0 0 4zm14 0a2 2 0 1 0 0-4 2 2 0 0 0 0 4zm-7-14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z' },
    { id: 'pie', label: 'Pie Chart', icon: 'M21.21 15.89A10 10 0 1 1 8 2.83M22 12A10 10 0 0 0 12 2v10z' },
    { id: 'box', label: 'Box Plot', icon: 'M12 3v18M5 9h14M5 15h14' },
];

export default function ChartBuilderPage() {
    const [session, setSession] = useState<any>(null);
    const [columns, setColumns] = useState<string[]>([]);
    const [config, setConfig] = useState<any>({ type: 'bar', x: '', y: '', color: 'None' });
    const [chartData, setChartData] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [suggestions, setSuggestions] = useState<any[]>([]);

    useEffect(() => {
        init();
    }, []);

    const init = async () => {
        try {
            const sessionData = await api.getSession();
            setSession(sessionData);
            if (sessionData.has_data) {
                setColumns(sessionData.column_names || []);
                // Load suggestions
                const recs = await api.getRecommendations();
                setSuggestions(recs.recommendations || []);
            }
        } catch (err) {
            console.error(err);
        }
    };

    const updateConfig = (key: string, value: string) => {
        setConfig(prev => ({ ...prev, [key]: value }));
    };

    const handleDragStart = (e: React.DragEvent, col: string) => {
        e.dataTransfer.setData('text/plain', col);
    };

    const handleDrop = (e: React.DragEvent, zone: string) => {
        e.preventDefault();
        const col = e.dataTransfer.getData('text/plain');
        updateConfig(zone, col);
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
    };

    const generateChart = async () => {
        if (!config.x && !config.y && config.type !== 'pie') return;
        if (config.type === 'pie' && !config.column) {
            // Special handling for pie generic config mapping
            if (config.x) config.column = config.x;
        }

        setLoading(true);
        try {
            const res = await api.createCustomViz(config.type, config);
            if (res.success) {
                setChartData(res.data);
            }
        } catch (err) {
            console.error("Chart generation failed", err);
        } finally {
            setLoading(false);
        }
    };

    // Auto-generate when valid config changes
    useEffect(() => {
        if ((config.x && config.y) || (config.type === 'pie' && config.x)) {
            generateChart();
        }
    }, [config]);

    if (!session?.has_data) return <div className="p-8 text-center">No dataset loaded.</div>;

    return (
        <div className="container mx-auto px-4 py-8 h-[calc(100vh-100px)] flex gap-6">

            {/* Sidebar: Columns & Suggestions */}
            <div className="w-1/4 flex flex-col gap-6">
                {/* Columns Draggable List */}
                <div className="glass-card p-4 flex-grow flex flex-col h-1/2">
                    <h3 className="font-semibold text-gray-700 mb-2 text-sm uppercase tracking-wide">Fields</h3>
                    <p className="text-xs text-gray-400 mb-3">Drag fields to the configuration zones</p>
                    <div className="flex-grow overflow-y-auto custom-scrollbar space-y-2">
                        {columns.map(col => (
                            <div
                                key={col}
                                draggable
                                onDragStart={(e) => handleDragStart(e, col)}
                                className="bg-white p-2 rounded shadow-sm border border-gray-100 cursor-grab active:cursor-grabbing hover:bg-gray-50 text-sm font-medium text-gray-700 flex items-center justify-between group"
                            >
                                {col}
                                <svg className="text-gray-300 group-hover:text-gray-400" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="9" cy="12" r="1" /><circle cx="9" cy="5" r="1" /><circle cx="9" cy="19" r="1" /><circle cx="15" cy="12" r="1" /><circle cx="15" cy="5" r="1" /><circle cx="15" cy="19" r="1" /></svg>
                            </div>
                        ))}
                    </div>
                </div>

                {/* AI Suggestions List */}
                <div className="glass-card p-4 h-1/2 flex flex-col">
                    <h3 className="font-semibold text-gray-700 mb-3 text-sm uppercase tracking-wide flex items-center gap-2">
                        <svg className="text-primary-500" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" /></svg>
                        AI Suggestions
                    </h3>
                    <div className="flex-grow overflow-y-auto custom-scrollbar space-y-2">
                        {suggestions.map((rec, i) => (
                            <div
                                key={i}
                                onClick={() => {
                                    api.createVisualization(rec.type, rec).then(res => {
                                        if (res.success) setChartData(res.data);
                                    });
                                }}
                                className="p-3 bg-primary-50 rounded-lg cursor-pointer hover:bg-primary-100 transition-colors border border-primary-100"
                            >
                                <div className="text-xs font-bold text-primary-800 mb-1">{rec.type.toUpperCase()}</div>
                                <div className="text-xs text-primary-700 leading-tight line-clamp-2">{rec.description}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Main Builder Area */}
            <div className="w-3/4 flex flex-col gap-6">
                {/* Configuration Drop Zones */}
                <div className="glass-card p-4">
                    {/* Chart Type Selector */}
                    <div className="flex gap-4 mb-6 border-b border-gray-100 pb-4">
                        {CHART_TYPES.map(type => (
                            <button
                                key={type.id}
                                onClick={() => updateConfig('type', type.id)}
                                className={`flex flex-col items-center gap-2 px-4 py-2 rounded-lg transition-all ${config.type === type.id ? 'bg-primary-600 text-white shadow-md' : 'hover:bg-gray-50 text-gray-500'}`}
                            >
                                <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d={type.icon} /></svg>
                                <span className="text-xs font-medium">{type.label}</span>
                            </button>
                        ))}
                    </div>

                    {/* Axis Zones */}
                    <div className="grid grid-cols-3 gap-6">
                        <div
                            onDrop={(e) => handleDrop(e, 'x')}
                            onDragOver={handleDragOver}
                            className={`border-2 border-dashed rounded-xl p-4 flex flex-col items-center justify-center min-h-[100px] transition-colors ${config.x ? 'border-primary-500 bg-primary-50' : 'border-gray-300 bg-gray-50 hover:border-primary-300'}`}
                        >
                            <span className="text-sm font-semibold text-gray-500 mb-1">X-Axis</span>
                            {config.x ? (
                                <div className="px-3 py-1 bg-white rounded shadow-sm text-primary-700 font-medium flex items-center gap-2">
                                    {config.x}
                                    <button onClick={(e) => { e.stopPropagation(); updateConfig('x', ''); }} className="hover:text-red-500">×</button>
                                </div>
                            ) : <span className="text-xs text-gray-400">Drop Field Here</span>}
                        </div>

                        <div
                            onDrop={(e) => handleDrop(e, 'y')}
                            onDragOver={handleDragOver}
                            className={`border-2 border-dashed rounded-xl p-4 flex flex-col items-center justify-center min-h-[100px] transition-colors ${config.y ? 'border-primary-500 bg-primary-50' : 'border-gray-300 bg-gray-50 hover:border-primary-300'}`}
                        >
                            <span className="text-sm font-semibold text-gray-500 mb-1">Y-Axis / Measure</span>
                            {config.y ? (
                                <div className="px-3 py-1 bg-white rounded shadow-sm text-primary-700 font-medium flex items-center gap-2">
                                    {config.y}
                                    <button onClick={(e) => { e.stopPropagation(); updateConfig('y', ''); }} className="hover:text-red-500">×</button>
                                </div>
                            ) : <span className="text-xs text-gray-400">Drop Field Here</span>}
                        </div>

                        <div
                            onDrop={(e) => handleDrop(e, 'color')}
                            onDragOver={handleDragOver}
                            className={`border-2 border-dashed rounded-xl p-4 flex flex-col items-center justify-center min-h-[100px] transition-colors ${config.color !== 'None' ? 'border-primary-500 bg-primary-50' : 'border-gray-300 bg-gray-50 hover:border-primary-300'}`}
                        >
                            <span className="text-sm font-semibold text-gray-500 mb-1">Color / Group</span>
                            {config.color && config.color !== 'None' ? (
                                <div className="px-3 py-1 bg-white rounded shadow-sm text-primary-700 font-medium flex items-center gap-2">
                                    {config.color}
                                    <button onClick={(e) => { e.stopPropagation(); updateConfig('color', 'None'); }} className="hover:text-red-500">×</button>
                                </div>
                            ) : <span className="text-xs text-gray-400">Drop Field Here (Optional)</span>}
                        </div>
                    </div>
                </div>

                {/* Chart Preview */}
                <div className="glass-card p-6 flex-grow relative overflow-hidden flex flex-col">
                    {loading && (
                        <div className="absolute inset-0 bg-white/80 z-10 flex items-center justify-center backdrop-blur-sm">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
                        </div>
                    )}

                    {chartData ? (
                        <div className="flex-grow w-full h-full">
                            <Plot
                                data={chartData.data}
                                layout={{
                                    ...chartData.layout,
                                    autosize: true,
                                    paper_bgcolor: 'rgba(0,0,0,0)',
                                    plot_bgcolor: 'rgba(0,0,0,0)',
                                    font: { family: 'Segoe UI' },
                                    margin: { t: 40, r: 20, l: 60, b: 60 }
                                }}
                                style={{ width: '100%', height: '100%' }}
                                useResizeHandler
                                config={{ responsive: true }}
                            />
                        </div>
                    ) : (
                        <div className="flex-grow flex flex-col items-center justify-center text-gray-400 border-2 border-dashed border-gray-100 rounded-2xl">
                            <svg className="w-16 h-16 mb-4 text-gray-200" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" /></svg>
                            <p>Configure axes to generate a chart</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
