import React from 'react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  highlight?: boolean;
}

export const StatCard: React.FC<StatCardProps> = ({ 
  title, 
  value, 
  subtitle, 
  trend, 
  trendValue,
  highlight = false
}) => {
  const getTrendColor = () => {
    if (trend === 'up') return 'text-emerald-600';
    if (trend === 'down') return 'text-stanford';
    return 'text-slate-400';
  };

  return (
    <div className="bg-white p-6 rounded-sm shadow-sm border border-slate-100 flex flex-col justify-between h-32 relative overflow-hidden group hover:border-slate-300 transition-colors">
      <div className="flex justify-between items-start z-10">
          <h3 className="text-slate-500 text-xs font-bold uppercase tracking-widest">{title}</h3>
          {trendValue && (
            <span className={`text-xs font-semibold ${getTrendColor()} bg-slate-50 px-2 py-1 rounded-full`}>
                {trend === 'up' ? '↑' : '↓'} {trendValue}
            </span>
          )}
      </div>
      
      <div className="z-10">
        <div className="flex items-baseline gap-2">
            <span className={`font-serif text-4xl font-semibold tracking-tight ${highlight ? 'text-stanford' : 'text-slate-900'}`}>
                {value}
            </span>
        </div>
        {subtitle && <p className="mt-1 text-xs text-slate-400 font-medium">{subtitle}</p>}
      </div>

      {/* Decorative gradient blob */}
      <div className="absolute -bottom-4 -right-4 w-20 h-20 bg-slate-50 rounded-full group-hover:bg-slate-100 transition-colors -z-0"></div>
    </div>
  );
};