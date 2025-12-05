import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { YearlyStat, FramingLabel, COLORS } from '../types';

interface TrendChartProps {
  data: YearlyStat[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white p-3 border border-slate-200 shadow-lg rounded-sm min-w-[150px]">
        <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">{label}</p>
        <div className="space-y-1">
          {payload.map((p: any) => (
            <div key={p.name} className="flex items-center justify-between text-xs gap-4">
              <span style={{ color: p.color }} className="font-medium">{p.name}</span>
              <span className="font-mono text-slate-700">{p.value}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return null;
};

export const TrendChart: React.FC<TrendChartProps> = ({ data }) => {
  // Filter out 2025 as there's insufficient data
  const filteredData = data.filter(d => d.year !== 2025);

  const chartData = filteredData.map(d => ({
    year: d.year,
    Environment: d.Environment,
    Work: d.Work,
    CommunityAndLife: d.CommunityAndLife,
  }));

  return (
    <div className="h-full flex flex-col">
      <div className="mb-6">
        <div className="flex justify-between items-start">
          <div>
            <h2 className="font-sans text-sm font-bold text-slate-900 uppercase tracking-wide">Narrative Frame Trends</h2>
            <p className="text-xs text-slate-500 mt-1 font-medium">
              Longitudinal tracking of key strategic frames (2013–Present).
            </p>
          </div>
          <div className="bg-slate-50 px-2 py-1 border border-slate-100 rounded text-[10px] font-mono text-slate-400">
            AUTO-SCALE: ON
          </div>
        </div>
      </div>
      <div className="h-[400px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 10, right: 10, left: -10, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis
              dataKey="year"
              stroke="#94a3b8"
              fontSize={10}
              tickLine={false}
              axisLine={{ stroke: '#e2e8f0' }}
              tickMargin={10}
              fontFamily="Inter, sans-serif"
              fontWeight={500}
            />
            <YAxis
              stroke="#94a3b8"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              fontFamily="Inter, sans-serif"
              fontWeight={500}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ paddingTop: '20px', fontFamily: 'Inter, sans-serif', fontSize: '11px', fontWeight: 500, color: '#64748b' }}
              iconType="circle"
              iconSize={8}
            />

            <Line
              type="monotone"
              dataKey="Environment"
              stroke={COLORS[FramingLabel.Environment]}
              strokeWidth={2.5}
              dot={{ r: 2, fill: COLORS[FramingLabel.Environment], strokeWidth: 0 }}
              activeDot={{ r: 5, stroke: '#fff', strokeWidth: 2 }}
              name="Environment"
            />
            <Line
              type="monotone"
              dataKey="Work"
              stroke={COLORS[FramingLabel.Work]}
              strokeWidth={2}
              dot={false}
              name="Work"
            />
            <Line
              type="monotone"
              dataKey="CommunityAndLife"
              stroke={COLORS[FramingLabel.CommunityAndLife]}
              strokeWidth={1.5}
              strokeDasharray="3 3"
              dot={false}
              name="Community & Life"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};