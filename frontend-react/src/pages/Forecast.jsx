import React, { useState } from 'react';
import { api, endpoints } from '../services/api';
import { TrendingUp, ArrowRight } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Forecast = () => {
    const [prevSpend, setPrevSpend] = useState('');
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);

    const handlePredict = async () => {
        if (!prevSpend) return;
        setLoading(true);
        try {
            const res = await api.post(endpoints.predictSpending, null, {
                params: { prev_month_spend: parseFloat(prevSpend) }
            });
            setPrediction(res.data.predicted_spend);
        } catch (err) {
            console.error("Prediction Error", err);
            alert("Failed to get prediction");
        } finally {
            setLoading(false);
        }
    };

    const chartData = prediction ? [
        { name: 'Last Month', value: parseFloat(prevSpend) },
        { name: 'Next Month (Predicted)', value: prediction }
    ] : [];


    return (
        <div className="max-w-4xl mx-auto animation-fade-in space-y-8">
            <div>
                <h1 className="text-3xl font-bold text-white tracking-tight">AI Forecaster</h1>
                <p className="text-secondary mt-1">Predict your future spending based on historical data patterns.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Input Card */}
                <div className="bg-card p-8 rounded-xl border border-gray-800/60 shadow-lg flex flex-col justify-center">
                    <div className="mb-6">
                        <div className="w-12 h-12 bg-primary/20 rounded-lg flex items-center justify-center text-primary mb-4">
                            <TrendingUp size={24} />
                        </div>
                        <h2 className="text-xl font-bold text-white">Input Data</h2>
                        <p className="text-gray-400 text-sm mt-1">Enter your last month's total spend to generate a prediction.</p>
                    </div>

                    <div className="space-y-4">
                        <div>
                            <label className="block text-gray-400 text-sm font-medium mb-2">Previous Month Spend (₹)</label>
                            <div className="flex gap-3">
                                <input
                                    type="number"
                                    value={prevSpend}
                                    onChange={(e) => setPrevSpend(e.target.value)}
                                    placeholder="e.g. 25000"
                                    className="flex-1 bg-dark border border-gray-800 rounded-lg p-3 text-white placeholder-gray-600 focus:border-primary focus:ring-1 focus:ring-primary focus:outline-none transition-all"
                                />
                                <button
                                    onClick={handlePredict}
                                    disabled={loading || !prevSpend}
                                    className="bg-primary text-white font-semibold px-6 rounded-lg hover:bg-blue-600 transition-all shadow-lg shadow-blue-500/20 disabled:opacity-50"
                                >
                                    {loading ? 'Analyzing...' : 'Predict'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Results Card */}
                <div className={`bg-gradient-to-br from-gray-900 to-black p-1 rounded-xl border border-gray-800 shadow-xl transition-all duration-500 ${prediction ? 'opacity-100 translate-y-0' : 'opacity-50 translate-y-2'}`}>
                    <div className="bg-card h-full rounded-lg p-8 relative overflow-hidden">
                        {prediction ? (
                            <div className="relative z-10 h-full flex flex-col">
                                <h2 className="text-secondary text-sm font-semibold uppercase tracking-wider mb-1">Forecast Result</h2>
                                <div className="flex items-baseline gap-1 mb-6">
                                    <span className="text-4xl font-bold text-white">₹{prediction.toLocaleString()}</span>
                                    <span className="text-gray-500 text-sm font-medium">projected</span>
                                </div>

                                <div className="flex items-center gap-3 mb-8 p-3 bg-gray-800/50 rounded-lg border border-gray-700/50 w-fit">
                                    {parseFloat(prevSpend) < prediction ? (
                                        <div className="flex items-center text-red-400 text-sm font-bold">
                                            <TrendingUp size={16} className="mr-1.5" />
                                            +{((prediction - prevSpend) / prevSpend * 100).toFixed(1)}% Increase
                                        </div>
                                    ) : (
                                        <div className="flex items-center text-emerald-400 text-sm font-bold">
                                            <TrendingUp size={16} className="mr-1.5 rotate-180" />
                                            -{((prevSpend - prediction) / prevSpend * 100).toFixed(1)}% Decrease
                                        </div>
                                    )}
                                </div>

                                <div className="flex-1 min-h-[200px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={chartData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                                            <XAxis dataKey="name" stroke="#52525b" tick={{ fill: '#71717a', fontSize: 11 }} axisLine={false} tickLine={false} />
                                            <YAxis hide />
                                            <Tooltip
                                                cursor={{ fill: '#27272a' }}
                                                contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', color: '#fff' }}
                                            />
                                            <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} barSize={60} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        ) : (
                            <div className="h-full flex flex-col items-center justify-center text-center text-gray-500">
                                <div className="w-16 h-16 bg-gray-800/50 rounded-full flex items-center justify-center mb-4">
                                    <TrendingUp size={32} className="text-gray-600" />
                                </div>
                                <p>Enter a value to see AI predictions</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Forecast;
