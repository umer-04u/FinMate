import axios from 'axios';

// Use env override if provided, else same-origin (works on Render)
const API_URL = import.meta.env.VITE_API_BASE_URL || window.location.origin;

export const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const endpoints = {
    checkAnomaly: '/analyze/anomaly',
    predictSpending: '/predict/spending',
    categorize: '/categorize',
    getTransactions: '/transactions',
    addTransaction: '/transactions',
    chat: '/chat',
};

// Helper function to handle API errors
export const handleApiError = (error) => {
    console.error("API Error:", error);
    throw error;
};
