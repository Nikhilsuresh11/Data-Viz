import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Data-Viz
          </h1>
          <p className="text-xl text-gray-600">
            AI-Powered Data Analysis & Visualization
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
          <Link href="/upload" className="bg-white p-8 rounded-lg shadow-lg hover:shadow-xl transition-shadow">
            <div className="text-4xl mb-4">📊</div>
            <h2 className="text-2xl font-semibold mb-2">Upload</h2>
            <p className="text-gray-600">Upload your CSV or Excel file to get started</p>
          </Link>

          <Link href="/analyze" className="bg-white p-8 rounded-lg shadow-lg hover:shadow-xl transition-shadow">
            <div className="text-4xl mb-4">🔍</div>
            <h2 className="text-2xl font-semibold mb-2">Analyze</h2>
            <p className="text-gray-600">Get AI-powered insights about your data</p>
          </Link>

          <Link href="/visualize" className="bg-white p-8 rounded-lg shadow-lg hover:shadow-xl transition-shadow">
            <div className="text-4xl mb-4">📈</div>
            <h2 className="text-2xl font-semibold mb-2">Visualize</h2>
            <p className="text-gray-600">Create beautiful interactive charts</p>
          </Link>
        </div>

        <div className="mt-16 text-center text-sm text-gray-500">
          <p>Optimized for Render Free Tier • Lazy Loading • Memory Efficient</p>
        </div>
      </div>
    </div>
  );
}
