'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
    const pathname = usePathname();

    const isActive = (path: string) => {
        return pathname === path ? 'text-primary-600 font-semibold' : 'text-gray-500 hover:text-gray-900';
    };

    return (
        <nav className="bg-white/80 backdrop-blur-md border-b border-gray-100 sticky top-0 z-50">
            <div className="container mx-auto px-4">
                <div className="flex justify-between items-center h-16">
                    {/* Logo */}
                    <Link href="/" className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-600 to-emerald-500 flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-primary-500/30">
                            DV
                        </div>
                        <span className="font-bold text-xl text-gray-800 tracking-tight">Data<span className="text-primary-600">Viz</span></span>
                    </Link>

                    {/* Navigation Links */}
                    <div className="hidden md:flex items-center space-x-6">
                        <Link href="/" className={isActive('/')}>Home</Link>
                        <Link href="/upload" className={isActive('/upload')}>Upload</Link>
                        <Link href="/overview" className={isActive('/overview')}>Overview</Link>
                        <Link href="/explorer" className={isActive('/explorer')}>Explorer</Link>
                        <Link href="/visualize" className={isActive('/visualize')}>Visualize</Link>
                        <Link href="/chart-builder" className={isActive('/chart-builder')}>Chart Builder</Link>
                        <Link href="/insights" className={isActive('/insights')}>Insights</Link>
                    </div>

                    {/* CTA Button */}
                    <div className="flex items-center gap-4">
                        <Link href="/upload" className="hidden md:flex px-4 py-2 bg-gray-900 hover:bg-black text-white rounded-lg font-medium transition-colors shadow-lg shadow-gray-900/20 text-sm">
                            New Project
                        </Link>

                        {/* Mobile Menu Button  */}
                        <button className="md:hidden p-2 text-gray-600">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="18" x2="21" y2="18" /></svg>
                        </button>
                    </div>
                </div>
            </div>
        </nav>
    );
}
