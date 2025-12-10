/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Backgrounds: Zinc/Slate for a premium 'Enterprise Dark' feel
                dark: "#09090b",       // Main background (Zinc 950)
                card: "#18181b",       // Card background (Zinc 900)
                hover: "#27272a",      // Hover state

                // Accents
                primary: "#3b82f6",    // Professional Blue
                secondary: "#64748b",  // Slate text

                // Semantic
                success: "#10b981",    // Emerald
                danger: "#ef4444",     // Red
                warning: "#f59e0b",    // Amber
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
