import React, { useState } from 'react';
import { api, endpoints } from '../services/api';
import { useNavigate } from 'react-router-dom';
import { AlertTriangle, CheckCircle, Loader } from 'lucide-react';

const AddTransaction = () => {
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
        merchant: '',
        amount: '',
        date: new Date().toISOString().split('T')[0],
    });
    const [category, setCategory] = useState(null);
    const [anomalyCheck, setAnomalyCheck] = useState(null);
    const [loading, setLoading] = useState(false);
    const [submitting, setSubmitting] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
        // Reset checks if merchant changes
        if (e.target.name === 'merchant') {
            setCategory(null);
            setAnomalyCheck(null);
        }
    };

    const analyzeTransaction = async (e) => {
        e.preventDefault();
        if (!formData.merchant || !formData.amount) return;

        setLoading(true);
        try {
            // 1. Categorize
            const catRes = await api.post(endpoints.categorize, { merchant: formData.merchant });
            const detectedCategory = catRes.data.category;
            setCategory(detectedCategory);

            // 2. Check Anomaly
            const anomalyPayload = {
                amount: parseFloat(formData.amount),
                category: detectedCategory,
                date: formData.date,
                merchant: formData.merchant
            };
            const anomRes = await api.post(endpoints.checkAnomaly, anomalyPayload);
            setAnomalyCheck(anomRes.data);

        } catch (err) {
            console.error("Analysis Failed", err);
            alert("Failed to analyze transaction.");
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = async () => {
        if (!category || !formData.amount) return;
        setSubmitting(true);
        try {
            const payload = {
                amount: parseFloat(formData.amount),
                category: category,
                date: formData.date,
                merchant: formData.merchant
            };
            await api.post(endpoints.addTransaction, payload);
            navigate('/'); // Go back to dashboard on success
        } catch (err) {
            console.error("Save Failed", err);
            alert("Failed to save transaction.");
        } finally {
            setSubmitting(false);
        }
    };


    return (
        <div className="max-w-2xl mx-auto space-y-8 animate-fade-in">
            <div>
                <h1 className="text-3xl font-bold text-white tracking-tight">Add Transaction</h1>
                <p className="text-secondary mt-1">Log your expenses or use AI to auto-categorize them.</p>
            </div>

            <div className="bg-card p-8 rounded-xl border border-gray-800/60 shadow-lg relative overflow-hidden">
                <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl -mr-32 -mt-32 pointer-events-none"></div>

                <form onSubmit={analyzeTransaction} className="space-y-6 relative z-10">
                    <div>
                        <label className="block text-gray-400 text-sm font-medium mb-2">Merchant Name</label>
                        <input
                            type="text"
                            name="merchant"
                            value={formData.merchant}
                            onChange={handleChange}
                            placeholder="e.g. Starbucks, Uber, Netflix"
                            className="w-full bg-dark border border-gray-800 rounded-lg p-3 text-white placeholder-gray-600 focus:border-primary focus:ring-1 focus:ring-primary focus:outline-none transition-all"
                        />
                    </div>

                    <div className="grid grid-cols-2 gap-6">
                        <div>
                            <label className="block text-gray-400 text-sm font-medium mb-2">Amount (₹)</label>
                            <input
                                type="number"
                                name="amount"
                                value={formData.amount}
                                onChange={handleChange}
                                placeholder="0.00"
                                step="0.01"
                                className="w-full bg-dark border border-gray-800 rounded-lg p-3 text-white placeholder-gray-600 focus:border-primary focus:ring-1 focus:ring-primary focus:outline-none transition-all"
                            />
                        </div>
                        <div>
                            <label className="block text-gray-400 text-sm font-medium mb-2">Date</label>
                            <input
                                type="date"
                                name="date"
                                value={formData.date}
                                onChange={handleChange}
                                className="w-full bg-dark border border-gray-800 rounded-lg p-3 text-white focus:border-primary focus:ring-1 focus:ring-primary focus:outline-none transition-all"
                            />
                        </div>
                    </div>

                    {category && (
                        <div className="bg-gray-800/30 p-4 rounded-lg border border-gray-700/50 animate-fade-in">
                            <div className="flex justify-between items-start mb-4">
                                <div>
                                    <p className="text-gray-500 text-xs uppercase tracking-wider font-semibold">AI Detected Category</p>
                                    <p className="text-xl font-bold text-white mt-1">{category}</p>
                                </div>
                                <span className="px-2 py-1 bg-primary/10 text-primary text-xs rounded-full font-medium border border-primary/20">
                                    High Confidence
                                </span>
                            </div>

                            {anomalyCheck?.is_anomaly ? (
                                <div className="flex items-center gap-3 text-red-400 bg-red-400/5 p-3 rounded-lg border border-red-400/10">
                                    <AlertTriangle size={18} />
                                    <div>
                                        <p className="text-sm font-semibold">Unusual Spend Detected!</p>
                                        <p className="text-xs opacity-80">This amount is higher than your average for {category}.</p>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex items-center gap-3 text-emerald-400 bg-emerald-400/5 p-3 rounded-lg border border-emerald-400/10">
                                    <CheckCircle size={18} />
                                    <div>
                                        <p className="text-sm font-semibold">Within Normal Range</p>
                                        <p className="text-xs opacity-80">This transaction looks consistent with history.</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    <div className="pt-2 flex gap-4">
                        {!category ? (
                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full bg-primary text-white font-semibold py-3 rounded-lg hover:bg-blue-600 transition-all shadow-lg shadow-blue-500/20 disabled:opacity-50 flex justify-center items-center gap-2"
                            >
                                {loading ? <Loader className="animate-spin" size={18} /> : 'AI Analyze & Categorize'}
                            </button>
                        ) : (
                            <>
                                <button
                                    type="button"
                                    onClick={() => setCategory(null)}
                                    className="flex-1 bg-transparent border border-gray-700 text-gray-400 font-medium py-3 rounded-lg hover:bg-gray-800 hover:text-white transition-all"
                                >
                                    Edit
                                </button>
                                <button
                                    type="button"
                                    onClick={handleSubmit}
                                    disabled={submitting}
                                    className="flex-[2] bg-emerald-500 text-white font-semibold py-3 rounded-lg hover:bg-emerald-600 transition-all shadow-lg shadow-emerald-500/20 disabled:opacity-50 flex justify-center items-center gap-2"
                                >
                                    {submitting ? <Loader className="animate-spin" size={18} /> : 'Confirm Transaction'}
                                </button>
                            </>
                        )}
                    </div>
                </form>
            </div>
        </div>
    );
};

export default AddTransaction;
