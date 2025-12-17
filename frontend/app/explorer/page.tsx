'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api';

export default function ExplorerPage() {
    const [loading, setLoading] = useState(true);
    const [session, setSession] = useState<any>(null);
    const [columns, setColumns] = useState<string[]>([]);
    const [tableData, setTableData] = useState<any[]>([]);
    const [totalRows, setTotalRows] = useState(0);

    // Filtering state
    const [filters, setFilters] = useState<any>({});
    const [activeColumn, setActiveColumn] = useState<string | null>(null);
    const [columnStats, setColumnStats] = useState<any>(null); // For filter sidebar

    // Pagination (basic client-side simulation for demo, real implementation would need server pagination params)
    const [page, setPage] = useState(1);
    const ROWS_per_PAGE = 50;

    useEffect(() => {
        init();
    }, []);

    const init = async () => {
        try {
            const sessionData = await api.getSession();
            setSession(sessionData);

            if (sessionData.has_data) {
                setColumns(sessionData.column_names || []);
                // Initial load with no filters
                handleFilter({});
            }
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleFilter = async (newFilters: any) => {
        setLoading(true);
        try {
            const res = await api.filterData(newFilters);
            if (res.success) {
                setTableData(res.sample || []); // Backend returns 'sample' which is currently top 10 rows, need to adjust backend for pagination or use what we have
                setTotalRows(res.rows || 0);
                setFilters(newFilters);
            }
        } catch (err) {
            console.error("Filter failed", err);
        } finally {
            setLoading(false);
        }
    };

    const handleColumnSelect = async (col: string) => {
        setActiveColumn(col);
        try {
            const stats = await api.getColumnData(col);
            setColumnStats(stats);
        } catch (err) {
            console.error(err);
        }
    };

    const downloadFiltered = async () => {
        try {
            const blob = await api.exportData(filters);
            const url = window.URL.createObjectURL(new Blob([blob]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'filtered_data.csv');
            document.body.appendChild(link);
            link.click();
            if (link.parentNode) link.parentNode.removeChild(link);
        } catch (err) {
            console.error("Export failed", err);
            alert("Failed to download data");
        }
    };

    if (!session?.has_data) return <div className="p-8 text-center text-gray-500">No data loaded.</div>;

    return (
        <div className="container mx-auto px-4 py-8 h-[calc(100vh-100px)] flex flex-col">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-3xl font-bold text-gray-800">Data Explorer</h1>
                    <p className="text-gray-500">{totalRows.toLocaleString()} rows visible</p>
                </div>
                <div className="flex gap-2">
                    <button onClick={() => handleFilter({})} className="px-4 py-2 text-sm text-gray-600 hover:text-primary-600 transition-colors">
                        Reset Filters
                    </button>
                    <button onClick={downloadFiltered} className="btn-primary flex items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
                        Download CSV
                    </button>
                </div>
            </div>

            <div className="flex gap-6 flex-grow overflow-hidden">
                {/* Sidebar - Column List & Filter */}
                <div className="w-1/4 flex flex-col gap-4">
                    <div className="glass-card p-4 flex-grow flex flex-col overflow-hidden">
                        <h3 className="font-semibold text-gray-700 mb-3">Columns</h3>
                        <div className="flex-grow overflow-y-auto custom-scrollbar space-y-1">
                            {columns.map(col => (
                                <div
                                    key={col}
                                    onClick={() => handleColumnSelect(col)}
                                    className={`p-2 rounded cursor-pointer text-sm flex justify-between items-center transition-colors ${activeColumn === col ? 'bg-primary-50 text-primary-700 font-medium' : 'hover:bg-gray-50 text-gray-600'}`}
                                >
                                    <span className="truncate">{col}</span>
                                    {filters[col] && <span className="w-2 h-2 rounded-full bg-primary-500"></span>}
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Active Column Filter Stats */}
                    {activeColumn && (
                        <div className="glass-card p-4 h-1/3 overflow-y-auto">
                            <h3 className="font-semibold text-gray-800 mb-2 truncate">{activeColumn}</h3>
                            {columnStats ? (
                                <div className="space-y-3">
                                    <div className="text-xs text-gray-500 uppercase tracking-wide">Statistics</div>
                                    {columnStats.stats.mean !== undefined ? (
                                        <div className="text-sm grid grid-cols-2 gap-2">
                                            <div className="bg-gray-50 p-2 rounded">Min: <span className="font-medium text-gray-900">{columnStats.stats.min.toFixed(2)}</span></div>
                                            <div className="bg-gray-50 p-2 rounded">Max: <span className="font-medium text-gray-900">{columnStats.stats.max.toFixed(2)}</span></div>
                                            <div className="bg-gray-50 p-2 rounded col-span-2">Mean: <span className="font-medium text-gray-900">{columnStats.stats.mean.toFixed(2)}</span></div>
                                        </div>
                                    ) : (
                                        <div className="text-sm space-y-2">
                                            <div className="bg-gray-50 p-2 rounded flex justify-between">Unique Values: <span className="font-medium text-gray-900">{columnStats.stats.unique}</span></div>
                                            <div className="bg-gray-50 p-2 rounded flex justify-between">Most Common: <span className="font-medium text-gray-900 truncation max-w-[100px]">{columnStats.stats.top}</span></div>
                                        </div>
                                    )}

                                    {/* Simple Filter UI Placeholder */}
                                    <div className="pt-2 border-t border-gray-100">
                                        <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">Filter</div>
                                        {/* Range slider or checkbox list would go here */}
                                        <div className="text-xs text-gray-400 italic">Filter controls coming soon</div>
                                    </div>
                                </div>
                            ) : (
                                <div className="animate-pulse space-y-2">
                                    <div className="h-4 bg-gray-100 rounded w-3/4"></div>
                                    <div className="h-20 bg-gray-100 rounded w-full"></div>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Main Data Grid */}
                <div className="w-3/4 glass-card p-0 overflow-hidden flex flex-col">
                    <div className="overflow-auto flex-grow custom-scrollbar">
                        <table className="w-full text-left border-collapse">
                            <thead className="bg-gray-50 sticky top-0 z-10">
                                <tr>
                                    {columns.map(col => (
                                        <th key={col} className="p-4 text-xs font-semibold text-gray-500 uppercase tracking-wider border-b border-gray-200 whitespace-nowrap min-w-[150px]">
                                            {col}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-100">
                                {loading ? (
                                    <tr><td colSpan={columns.length} className="p-8 text-center text-gray-500">Loading data...</td></tr>
                                ) : (
                                    tableData.map((row, i) => (
                                        <tr key={i} className="hover:bg-gray-50 transition-colors">
                                            {columns.map(col => (
                                                <td key={`${i}-${col}`} className="p-4 text-sm text-gray-700 border-r border-transparent last:border-r-0 whitespace-nowrap max-w-[200px] overflow-hidden text-ellipsis">
                                                    {row[col] !== null ? String(row[col]) : ''}
                                                </td>
                                            ))}
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                    <div className="bg-white border-t border-gray-100 p-4 text-xs text-gray-500 flex justify-between items-center">
                        <span>Showing top {tableData.length} of {totalRows} rows</span>
                        {/* Pagination controls would go here */}
                        <div className="flex gap-2">
                            <button className="px-3 py-1 bg-gray-100 rounded hover:bg-gray-200 disabled:opacity-50" disabled>Previous</button>
                            <button className="px-3 py-1 bg-gray-100 rounded hover:bg-gray-200 disabled:opacity-50" disabled>Next</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
