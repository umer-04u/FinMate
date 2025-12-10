import React from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#ec4899'];
const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-gray-900 p-3 border border-gray-700 rounded-lg shadow-xl backdrop-blur-sm bg-opacity-90">
                <p className="font-bold text-white mb-1">{payload[0].name}</p>
                <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: payload[0].fill }}></span>
                    <p className="text-gray-300">₹{payload[0].value.toFixed(2)}</p>
                </div>
            </div>
        );
    }
    return null;
};

const SpendingPieChart = ({ data }) => {
    return (
        <div className="h-[400px] w-full bg-card p-6 rounded-xl border border-gray-800/60 shadow-sm flex flex-col">
            <h3 className="text-lg font-semibold text-white mb-6">Expenses by Category</h3>
            {data.length === 0 ? (
                <div className="h-full flex items-center justify-center text-gray-500">No data available</div>
            ) : (
                <div className="flex-1 min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={data}
                                cx="50%"
                                cy="50%"
                                innerRadius={80}
                                outerRadius={110}
                                fill="#8884d8"
                                paddingAngle={5}
                                dataKey="value"
                                stroke="none"
                            >
                                {data.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip content={<CustomTooltip />} />
                            <Legend
                                verticalAlign="bottom"
                                height={36}
                                iconType="circle"
                                wrapperStyle={{ paddingTop: '20px' }}
                            />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
            )}
        </div>
    );
};

export default SpendingPieChart;
