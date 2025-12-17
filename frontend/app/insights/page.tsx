'use client';

import { useState, useEffect, useRef } from 'react';
import { api } from '@/lib/api';

export default function InsightsPage() {
    const [session, setSession] = useState<any>(null);
    const [insights, setInsights] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    // Chat state
    const [messages, setMessages] = useState<any[]>([
        { role: 'assistant', content: "Hello! I've analyzed your dataset. Ask me anything about trends, correlations, or outliers." }
    ]);
    const [input, setInput] = useState('');
    const [chatLoading, setChatLoading] = useState(false);
    const messagesEndRef = useRef<null | HTMLDivElement>(null);

    useEffect(() => {
        init();
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const init = async () => {
        try {
            const sessionData = await api.getSession();
            setSession(sessionData);
            if (sessionData.has_data) {
                const insightData = await api.getInsights();
                setInsights(insightData);
            }
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleSend = async (e?: React.FormEvent) => {
        e?.preventDefault();
        if (!input.trim()) return;

        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setChatLoading(true);

        try {
            const res = await api.chat(userMsg.content);
            if (res.response) {
                setMessages(prev => [...prev, { role: 'assistant', content: res.response }]);
            } else if (res.error) {
                setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${res.error}` }]);
            } else {
                setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, I encountered an error analyzing that query." }]);
            }
        } catch (err) {
            console.error("Chat error:", err);
            setMessages(prev => [...prev, { role: 'assistant', content: "Network error. Please try again." }]);
        } finally {
            setChatLoading(false);
        }
    };

    if (!session?.has_data) return <div className="p-8 text-center">No dataset loaded.</div>;

    return (
        <div className="container mx-auto px-4 py-8 h-[calc(100vh-100px)] flex gap-6">

            {/* Left Panel: Static Insights */}
            <div className="w-1/3 flex flex-col gap-6 overflow-hidden">
                <div className="glass-card p-6 flex-grow flex flex-col overflow-hidden">
                    <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                        <svg className="text-yellow-500" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a10 10 0 1 0 10 10 4 4 0 0 1-5-5 4 4 0 0 1-5-5" /><path d="M8.5 8.5v.01" /><path d="M16 16v.01" /><path d="M12 12v.01" /><path d="M12 17v.01" /><path d="M12 7v.01" /></svg>
                        Key Findings
                    </h2>

                    {loading ? (
                        <div className="space-y-4 animate-pulse">
                            <div className="h-20 bg-gray-100 rounded-lg"></div>
                            <div className="h-20 bg-gray-100 rounded-lg"></div>
                            <div className="h-20 bg-gray-100 rounded-lg"></div>
                        </div>
                    ) : (
                        <div className="flex-grow overflow-y-auto custom-scrollbar space-y-4 pr-2">
                            {insights?.insights?.map((insight: string, i: number) => (
                                <div key={i} className="p-4 bg-white border border-gray-100 rounded-xl shadow-sm hover:shadow-md transition-shadow">
                                    <div className="flex items-start gap-3">
                                        <span className="text-primary-500 font-bold text-lg mt-[-2px]">•</span>
                                        <p className="text-gray-700 text-sm leading-relaxed">{insight}</p>
                                    </div>
                                </div>
                            ))}

                            {/* Top Correlations Section */}
                            {insights?.top_correlations && insights.top_correlations.length > 0 && (
                                <div className="mt-6 pt-4 border-t border-gray-100">
                                    <h3 className="font-semibold text-gray-700 mb-3 text-sm uppercase tracking-wide">Top Correlations</h3>
                                    <div className="space-y-2">
                                        {insights.top_correlations.map(([cols, val]: [string[], number], i: number) => (
                                            <div key={i} className="flex justify-between items-center text-sm p-2 bg-gray-50 rounded">
                                                <span className="text-gray-600 truncate max-w-[70%] text-xs" title={`${cols[0]} & ${cols[1]}`}>
                                                    {cols[0]} & {cols[1]}
                                                </span>
                                                <span className={`font-mono font-semibold ${Math.abs(val) > 0.7 ? 'text-green-600' : 'text-blue-600'}`}>
                                                    {val.toFixed(2)}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* Right Panel: Chat Interface */}
            <div className="w-2/3 glass-card flex flex-col overflow-hidden p-0 relative">
                {/* Chat Header */}
                <div className="p-4 border-b border-gray-100 bg-white/50 backdrop-blur-sm z-10 flex justify-between items-center">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-primary-500 to-emerald-400 flex items-center justify-center text-white shadow-lg">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>
                        </div>
                        <div>
                            <h3 className="font-bold text-gray-800">Data Assistant</h3>
                            <div className="flex items-center gap-1.5">
                                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                                <span className="text-xs text-gray-500 font-medium">Online & Ready</span>
                            </div>
                        </div>
                    </div>
                    <button onClick={() => setMessages([messages[0]])} className="text-xs text-gray-400 hover:text-red-500 transition-colors">
                        Clear Chat
                    </button>
                </div>

                {/* Messages Area */}
                <div className="flex-grow overflow-y-auto p-6 space-y-6 bg-gray-50/50 scroll-smooth">
                    {messages.map((msg, i) => (
                        <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[80%] rounded-2xl p-4 shadow-sm ${msg.role === 'user'
                                ? 'bg-primary-600 text-white rounded-br-none'
                                : 'bg-white text-gray-800 border border-gray-100 rounded-bl-none'
                                }`}>
                                <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                            </div>
                        </div>
                    ))}
                    {chatLoading && (
                        <div className="flex justify-start">
                            <div className="bg-white rounded-2xl p-4 shadow-sm border border-gray-100 rounded-bl-none flex items-center gap-2">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-75"></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-150"></div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="p-4 bg-white border-t border-gray-100">
                    <form onSubmit={handleSend} className="relative">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask about your data..."
                            className="w-full pl-6 pr-14 py-4 bg-gray-50 border border-gray-200 rounded-full focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all shadow-inner text-gray-700 placeholder-gray-400"
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || chatLoading}
                            className="absolute right-2 top-2 p-2 bg-primary-600 text-white rounded-full hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" /></svg>
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}
