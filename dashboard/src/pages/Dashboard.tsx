import React, { useState, useEffect, useMemo } from 'react';
import { loadDataset, getOverallDistribution } from '../services/dataService';
import { TrendChart } from '../components/TrendChart';
import { FramingDistributionChart } from '../components/FramingDistributionChart';
import { StatCard } from '../components/StatCard';
import { YearlyStat, FramingLabel } from '../types';
import { Link } from 'react-router-dom';

// Icons
const ActivityIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
);
const GlobeIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>
);

const InfoIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="16" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12.01" y2="8"></line>
    </svg>
);

export const Dashboard: React.FC = () => {
    const [yearlyStats, setYearlyStats] = useState<YearlyStat[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isScrolled, setIsScrolled] = useState(false);

    // Load Data
    useEffect(() => {
        const loadData = async () => {
            try {
                setLoading(true);
                const stats = await loadDataset(`${import.meta.env.BASE_URL}data/yt_video.all.jsonl`);
                setYearlyStats(stats);

                // Optional: Log for verification
                console.log('Dataset loaded:', {
                    totalVideos: stats.reduce((sum, s) => sum + s.totalVideos, 0),
                    yearRange: `${stats[0]?.year} - ${stats[stats.length - 1]?.year}`,
                    years: stats.length
                });
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load dataset');
                console.error('Data loading error:', err);
            } finally {
                setLoading(false);
            }
        };

        loadData();

        const handleScroll = () => {
            setIsScrolled(window.scrollY > 10);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    // Calculate Global Aggregate Profile
    const globalProfile = useMemo(() => {
        if (yearlyStats.length === 0) return null;

        const overallDist = getOverallDistribution(yearlyStats);

        // Filter to only include the requested frames for the display profile
        const displayDistribution = {
            [FramingLabel.Environment]: overallDist.Environment,
            [FramingLabel.Work]: overallDist.Work,
            [FramingLabel.CommunityAndLife]: overallDist.CommunityAndLife,
        } as Record<FramingLabel, number>;

        const totalLabels = overallDist.Environment + overallDist.GreenInnovation +
            overallDist.EconomyAndBusiness + overallDist.Work +
            overallDist.CommunityAndLife + overallDist.Patriotism;

        const envShare = totalLabels > 0
            ? Math.round((overallDist.Environment / totalLabels) * 100)
            : 0;

        return {
            name: "Global Energy Sector",
            totalVideos: overallDist.totalVideos,
            distribution: displayDistribution,
            envShare: envShare
        };
    }, [yearlyStats]);

    // Aggregated Stats for Cards
    const overallStats = useMemo(() => {
        if (yearlyStats.length === 0) {
            return {
                totalVideos: 0,
                envVideosCount: 0,
                envShareGlobal: 0,
                workVideosCount: 0,
                communityVideosCount: 0
            };
        }

        const totals = getOverallDistribution(yearlyStats);
        const totalVideos = totals.totalVideos;

        // Note: These are label counts, not unique video counts
        // A single video can have multiple labels
        const envVideosCount = totals.Environment;
        const workVideosCount = totals.Work;
        const communityVideosCount = totals.CommunityAndLife;

        const totalLabels = totals.Environment + totals.GreenInnovation +
            totals.EconomyAndBusiness + totals.Work +
            totals.CommunityAndLife + totals.Patriotism;

        const envShareGlobal = totalLabels > 0
            ? Math.round((envVideosCount / totalLabels) * 100)
            : 0;

        return {
            totalVideos,
            envVideosCount,
            envShareGlobal,
            workVideosCount,
            communityVideosCount
        };
    }, [yearlyStats]);

    // Loading state
    if (loading) {
        return (
            <div className="min-h-screen bg-paper flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-stanford mx-auto mb-4"></div>
                    <p className="text-slate-600 font-medium">Loading dataset...</p>
                </div>
            </div>
        );
    }

    // Error state
    if (error) {
        return (
            <div className="min-h-screen bg-paper flex items-center justify-center">
                <div className="bg-white p-8 rounded-sm border border-red-200 max-w-md">
                    <h2 className="text-red-600 font-bold text-lg mb-2">Error Loading Dataset</h2>
                    <p className="text-slate-600 mb-4">{error}</p>
                    <button
                        onClick={() => window.location.reload()}
                        className="bg-slate-900 text-white px-4 py-2 rounded-sm text-sm font-medium hover:bg-slate-800"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-paper text-slate-900 font-sans selection:bg-stanford selection:text-white">

            {/* Institutional Strip */}
            <div className="bg-slate-900 text-white border-b border-slate-800 py-2 px-4 sm:px-6 lg:px-8">
                <div className="max-w-7xl mx-auto flex justify-between items-center text-[10px] sm:text-xs font-medium tracking-widest uppercase">
                    <div className="flex items-center space-x-4 sm:space-x-6">
                        <div className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-princeton"></span>
                            <span>Princeton</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-stanford"></span>
                            <span>Stanford</span>
                        </div>
                        <div className="hidden sm:flex items-center gap-2 text-slate-400">
                            <span className="w-1.5 h-1.5 rounded-full bg-white"></span>
                            <span>Hitachi</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Tool Header */}
            <header className={`sticky top-0 z-50 transition-all duration-300 ${isScrolled ? 'bg-white/95 backdrop-blur-md shadow-sm border-b border-slate-200 py-3' : 'bg-white border-b border-slate-200 py-6'}`}>
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                        <div className="flex items-center gap-3">
                            <div className="bg-slate-900 p-2 rounded-sm text-white">
                                <GlobeIcon />
                            </div>
                            <div>
                                <h1 className="font-sans font-bold text-slate-900 leading-none text-lg tracking-tight uppercase">
                                    Energy Sector Ad Monitor
                                </h1>
                                <p className="text-xs text-slate-500 font-medium tracking-wide mt-1">
                                    Multimodal Frame Tracking & Narrative Analysis
                                </p>
                            </div>
                        </div>

                        {/* Navigation Links */}
                        <div className="flex items-center gap-3">
                            <Link
                                to="/references"
                                className="flex items-center gap-2 text-xs font-medium text-slate-600 hover:text-stanford transition-colors px-3 py-2 rounded-sm hover:bg-slate-50"
                            >
                                <InfoIcon />
                                <span>References & Disclaimers</span>
                            </Link>
                            <div className="text-[10px] font-mono text-slate-400 uppercase tracking-widest">
                                Scope: Aggregate Sector Analysis
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

                {/* Intelligence Brief */}
                <div className="mb-10 grid grid-cols-1 lg:grid-cols-4 gap-6">
                    <div className="lg:col-span-3 bg-white p-6 rounded-sm border border-slate-200 shadow-sm relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-1 h-full bg-stanford"></div>
                        <h2 className="font-sans text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">Sector Intelligence Brief</h2>
                        <p className="font-serif text-lg text-slate-800 leading-relaxed max-w-3xl">
                            Aggregate analysis of <strong style={{ color: '#8C1515', fontWeight: 600 }}>10 hours</strong> of video advertisements reveals a shift in framing. Recent data indicates a convergence of all three narrative frames, with <strong style={{ color: '#0f172a' }}>"Community & Life"</strong>, <strong style={{ color: '#0f172a' }}>"Environment"</strong>, and <strong style={{ color: '#0f172a' }}>"Work"</strong> reaching near-equal prominence after years of divergent trajectories.</p>
                    </div>

                    <div className="lg:col-span-1 bg-slate-900 p-6 rounded-sm border border-slate-800 text-slate-300 flex flex-col justify-center">
                        <div className="flex items-center gap-2 mb-2 text-emerald-400">
                            <ActivityIcon />
                            <span className="text-xs font-bold uppercase tracking-widest">Active Monitoring</span>
                        </div>
                        <div className="text-2xl font-mono text-white font-bold">50+ Entities</div>
                        <div className="text-xs text-slate-500 mt-1">Real-time cross-platform ingestion</div>
                    </div>
                </div>

                {/* Dashboard Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 mb-12">

                    {/* KPI Column */}
                    <div className="lg:col-span-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        <StatCard
                            title="Total Volume"
                            value={overallStats.totalVideos.toLocaleString()}
                            subtitle="Assets Analyzed"
                        />
                        <StatCard
                            title="Environment Framing"
                            value={overallStats.envVideosCount.toLocaleString()}
                            subtitle="Environmental Narratives"
                        />
                        <StatCard
                            title="Work Framing"
                            value={overallStats.workVideosCount.toLocaleString()}
                            subtitle="Employment Narratives"
                        />
                        <StatCard
                            title="Community Framing"
                            value={overallStats.communityVideosCount.toLocaleString()}
                            subtitle="Life & Society Narratives"
                        />
                    </div>

                    {/* Main Viz Area */}
                    <div className="lg:col-span-3 space-y-8">
                        <div className="bg-white p-6 rounded-sm shadow-sm border border-slate-200 min-h-[500px]">
                            <TrendChart data={yearlyStats} />
                        </div>
                    </div>

                    {/* Sidebar Viz */}
                    <div className="lg:col-span-1">
                        {globalProfile && (
                            <div className="bg-white p-6 rounded-sm shadow-sm border border-slate-200 h-full">
                                <FramingDistributionChart data={globalProfile} />
                            </div>
                        )}
                    </div>

                </div>
            </main>
        </div>
    );
};

export default Dashboard;
