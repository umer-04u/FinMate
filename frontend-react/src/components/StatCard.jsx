import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

const StatCard = ({ title, value, icon, trend, className }) => {
    return (
        <div className={twMerge("p-6 bg-card rounded-xl border border-gray-800/60 shadow-sm flex items-start justify-between group hover:border-primary/50 transition-all duration-300 relative overflow-hidden", className)}>

            {/* Background Glow Effect */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none transition-opacity opacity-0 group-hover:opacity-100"></div>

            <div className="relative z-10">
                <h2 className="text-secondary text-xs font-semibold uppercase tracking-wider mb-2">{title}</h2>
                <p className="text-3xl font-bold text-white tracking-tight group-hover:text-primary transition-colors">{value}</p>

                {trend && (
                    <div className={clsx("flex items-center gap-1 mt-3 text-xs font-medium px-2 py-1 rounded-full w-fit",
                        trend === 'up' ? "bg-red-500/10 text-red-400" : "bg-emerald-500/10 text-emerald-400"
                    )}>
                        <span>{trend === 'up' ? '▲' : '▼'}</span>
                        <span>vs last month</span>
                    </div>
                )}
            </div>

            {icon && (
                <div className="p-3 bg-gray-800/50 rounded-lg text-gray-400 group-hover:bg-primary group-hover:text-white transition-all duration-300 shadow-inner">
                    {icon}
                </div>
            )}
        </div>
    );
};

export default StatCard;
