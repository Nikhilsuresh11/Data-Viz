import Link from 'next/link';

export default function Home() {
  return (
    <div>
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-dark-bg to-gray-800 text-white rounded-b-[2rem] shadow-xl relative overflow-hidden">
        <div className="absolute inset-0 bg-[url('/pattern.svg')] opacity-5"></div>
        <div className="container mx-auto px-4 py-24 text-center relative z-10">
          <h1 className="text-5xl md:text-6xl font-extrabold mb-6 animate-fade-in tracking-tight">
            Data Visualization <span className="text-primary-500">Made Simple</span>
          </h1>
          <p className="text-xl text-gray-300 mb-10 max-w-2xl mx-auto font-light leading-relaxed">
            Upload your data and get instant insights with intelligent visualizations powered by AI.
            Optimized for speed and efficiency.
          </p>
          <div className="flex justify-center gap-4">
            <Link
              href="/upload"
              className="px-8 py-4 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-semibold shadow-lg hover:shadow-primary-500/30 transition-all transform hover:-translate-y-1"
            >
              Get Started
            </Link>
            <Link
              href="#features"
              className="px-8 py-4 bg-white/10 hover:bg-white/20 text-white border border-white/20 rounded-lg font-semibold backdrop-blur-sm transition-all"
            >
              Learn More
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-gray-50">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Overview Card */}
            <div className="glass-card p-8 hover:-translate-y-2 transition-transform duration-300">
              <div className="w-14 h-14 bg-primary-100 rounded-lg flex items-center justify-center mb-6 text-primary-600">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" /><circle cx="12" cy="12" r="3" /></svg>
              </div>
              <h3 className="text-2xl font-bold mb-3 text-gray-800">Clear Overview</h3>
              <p className="text-gray-600 leading-relaxed">
                Get a comprehensive overview of your dataset with key metrics, missing value analysis, and type detection.
              </p>
            </div>

            {/* Smart viz Card */}
            <div className="glass-card p-8 hover:-translate-y-2 transition-transform duration-300">
              <div className="w-14 h-14 bg-primary-100 rounded-lg flex items-center justify-center mb-6 text-primary-600">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="20" x2="12" y2="10" /><line x1="18" y1="20" x2="18" y2="4" /><line x1="6" y1="20" x2="6" y2="16" /></svg>
              </div>
              <h3 className="text-2xl font-bold mb-3 text-gray-800">Smart Charts</h3>
              <p className="text-gray-600 leading-relaxed">
                Automatically generated interactive visualizations (Plotly) that highlight the most important aspects of your data.
              </p>
            </div>

            {/* AI Insights Card */}
            <div className="glass-card p-8 hover:-translate-y-2 transition-transform duration-300">
              <div className="w-14 h-14 bg-primary-100 rounded-lg flex items-center justify-center mb-6 text-primary-600">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a10 10 0 1 0 10 10h-5a5 5 0 0 1-5-5V2Z" /></svg>
              </div>
              <h3 className="text-2xl font-bold mb-3 text-gray-800">AI Insights</h3>
              <p className="text-gray-600 leading-relaxed">
                Unlock deep patterns and hidden relationships in your data using integrated LLM-powered analysis.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
