import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import { COLORS, FramingLabel } from '../types';

interface FramingDistributionChartProps {
  data: {
    name: string;
    distribution: Record<FramingLabel, number>;
  };
}

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white p-2 border border-slate-200 shadow-sm rounded-sm text-xs">
        <span className="font-semibold text-slate-700">{payload[0].payload.label}:</span>
        <span className="ml-2 font-mono">{payload[0].value} units</span>
      </div>
    );
  }
  return null;
};

export const FramingDistributionChart: React.FC<FramingDistributionChartProps> = ({ data }) => {
  const chartData = Object.entries(data.distribution).map(([label, count]) => ({
    label: label.replace('and', '&'), // Shorten for display
    count: count as number,
    color: COLORS[label as FramingLabel]
  })).sort((a, b) => (b.count as number) - (a.count as number));

  // Determine the most frequent label
  const topLabel = chartData[0]?.label || "N/A";

  return (
    <div className="h-full flex flex-col">
      <div className="mb-6 pb-4 border-b border-slate-100">
        <div className="flex flex-col gap-2">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Framing Distribution</span>
          <h2 className="font-serif text-xl font-bold text-slate-900 leading-none">{data.name}</h2>
          <div className="flex items-center gap-2 mt-1">
            <div className="text-[10px] font-bold px-2 py-0.5 rounded-sm uppercase tracking-wide bg-slate-100 text-slate-600 border border-slate-200">
              Primary: {topLabel}
            </div>
          </div>
        </div>
      </div>

      <div className="flex-grow w-full min-h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 0, right: 10, left: 0, bottom: 0 }}
            barGap={4}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
            <XAxis type="number" hide />
            <YAxis
              type="category"
              dataKey="label"
              width={100}
              tick={{ fontSize: 10, fill: '#64748b', fontFamily: 'Inter', fontWeight: 500 }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: '#f8fafc' }} />
            <Bar dataKey="count" radius={[0, 2, 2, 0]} barSize={12}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 pt-4 border-t border-slate-100">
        <p className="text-[10px] text-slate-400 text-center uppercase tracking-wider">
          Data Source: Multimodal Ingestion
        </p>
      </div>
    </div>
  );
};