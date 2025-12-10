import React, { useEffect, useState } from 'react';
import { api, endpoints } from '../services/api';
import StatCard from '../components/StatCard';
import SpendingPieChart from '../components/SpendingPieChart';
import MonthlyBarChart from '../components/MonthlyBarChart';
import { DollarSign, CreditCard, Wallet, AlertCircle } from 'lucide-react';

const Dashboard = () => {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await api.get(endpoints.getTransactions);
                setData(response.data);
            } catch (err) {
                setError("Failed to load dashboard data.");
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    // --- Calculations ---
    // Ensure we primarily look at expenses
    const expenses = data.filter(item => item.Is_Expense);

    // Total Lifetime Spend
    const totalSpend = expenses.reduce((acc, curr) => acc + curr.Amount, 0);

    // Avg Monthly Spend
    // Group by Month (Assume 'Month' column exists from CSV)
    const months = [...new Set(expenses.map(item => item.Month))];
    const avgMonthly = months.length ? totalSpend / months.length : 0;

    // Category Data
    const categoryMap = {};
    expenses.forEach(item => {
        if (!categoryMap[item.Category]) categoryMap[item.Category] = 0;
        categoryMap[item.Category] += item.Amount;
    });
    const categoryData = Object.keys(categoryMap).map(key => ({
        name: key,
        value: categoryMap[key]
    }));

    // Monthly Data
    const monthlyMap = {};
    expenses.forEach(item => {
        if (!monthlyMap[item.Month]) monthlyMap[item.Month] = 0;
        monthlyMap[item.Month] += item.Amount;
    });
    // Sort by date ideally, but for now simple string sort or just pass as is
    const monthlyData = Object.keys(monthlyMap).map(key => ({
        name: key,
        value: monthlyMap[key]
    })).sort((a, b) => a.name.localeCompare(b.name));

    if (loading) return <div className="p-8 text-white">Loading Dashboard...</div>;
    if (error) return <div className="p-8 text-red-500">{error}</div>;

    return (
        <div className="space-y-8">
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold text-white tracking-tight">Overview</h1>
                    <p className="text-secondary mt-1">Welcome back, here's what's happening with your finances today.</p>
                </div>
                <div className="flex gap-3">
                    <button className="px-4 py-2 bg-card border border-gray-800 text-gray-300 rounded-lg hover:bg-gray-800 transition-colors text-sm font-medium">
                        Last 30 Days
                    </button>
                    <button className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-blue-600 transition-colors shadow-lg shadow-blue-500/20 text-sm font-medium">
                        Download Report
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <StatCard
                    title="Total Lifetime Spend"
                    value={`₹${totalSpend.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`}
                    icon={<DollarSign size={24} />}
                    trend="up"
                />
                <StatCard
                    title="Avg Monthly Spend"
                    value={`₹${avgMonthly.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`}
                    icon={<CreditCard size={24} />}
                    trend="down"
                />
                <StatCard
                    title="Total Transactions"
                    value={data.length}
                    icon={<Wallet size={24} />}
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <SpendingPieChart data={categoryData} />
                <div className="bg-card p-6 rounded-xl border border-gray-800/60 shadow-sm">
                    <h3 className="text-lg font-semibold text-white mb-6">Monthly Trends</h3>
                    <div className="h-[300px]">
                        <MonthlyBarChart data={monthlyData} />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
