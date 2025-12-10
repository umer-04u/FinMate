import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-gray-900 p-3 border border-gray-700 rounded-lg shadow-xl backdrop-blur-sm bg-opacity-90">
                <p className="font-bold text-white mb-1">{label}</p>
                <p className="text-primary font-medium">₹{payload[0].value.toFixed(2)}</p>
            </div>
        );
    }
    return null;
};

const MonthlyBarChart = ({ data }) => {
    return (
        <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis
                    dataKey="name"
                    stroke="#52525b"
                    tick={{ fill: '#71717a', fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                />
                <YAxis
                    stroke="#52525b"
                    tick={{ fill: '#71717a', fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                />
                <Tooltip cursor={{ fill: '#27272a' }} content={<CustomTooltip />} />
                <Bar
                    dataKey="value"
                    fill="#3b82f6"
                    radius={[4, 4, 0, 0]}
                    maxBarSize={50}
                    activeBar={{ fill: '#60a5fa' }}
                />
            </BarChart>
        </ResponsiveContainer>
    );
};

export default MonthlyBarChart;
