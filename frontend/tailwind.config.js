/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        './pages/**/*.{js,ts,jsx,tsx,mdx}',
        './components/**/*.{js,ts,jsx,tsx,mdx}',
        './app/**/*.{js,ts,jsx,tsx,mdx}',
    ],
    theme: {
        extend: {
            colors: {
                primary: {
                    50: '#ecfdf5',
                    100: '#d1fae5',
                    200: '#a7f3d0',
                    300: '#6ee7b7',
                    400: '#34d399',
                    500: '#10b981', // Your #4CAF50 equivalent (approx)
                    600: '#059669', // Main Brand
                    700: '#047857', // Secondary #2E7D32 equivalent
                    800: '#065f46',
                    900: '#064e3b',
                },
                dark: {
                    bg: '#212529',
                    card: '#2b3035',
                }
            },
            fontFamily: {
                sans: ['Segoe UI', 'Tahoma', 'Geneva', 'Verdana', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
