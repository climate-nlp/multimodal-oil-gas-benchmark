import { FramingLabel, YearlyStat } from '../types';

interface RawVideoData {
    labels: string[];
    video_publish_date: string;
}

// Map string labels from dataset to enum
const labelMap: Record<string, FramingLabel> = {
    "Environment": FramingLabel.Environment,
    "Green Innovation": FramingLabel.GreenInnovation,
    "Economy and Business": FramingLabel.EconomyAndBusiness,
    "Work": FramingLabel.Work,
    "Community and Life": FramingLabel.CommunityAndLife,
    "Patriotism": FramingLabel.Patriotism
};

// Load and process dataset directly to yearly stats
// Note: Handles JSONL format (one JSON object per line)
export const loadDataset = async (filePath: string): Promise<YearlyStat[]> => {
    try {
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`Failed to load dataset: ${response.statusText}`);
        }
        const text = await response.text();

        // Parse JSONL format (one JSON object per line)
        const rawData: RawVideoData[] = text
            .trim()
            .split('\n')
            .filter(line => line.trim())
            .map(line => JSON.parse(line));

        return processYearlyStats(rawData);
    } catch (error) {
        console.error('Error loading dataset:', error);
        throw error;
    }
};

// Process yearly statistics directly from raw data
export const processYearlyStats = (rawData: RawVideoData[]): YearlyStat[] => {
    const stats: Record<number, YearlyStat> = {};

    rawData.forEach(video => {
        const date = new Date(video.video_publish_date);
        const year = date.getFullYear();

        if (!stats[year]) {
            stats[year] = {
                year,
                Environment: 0,
                GreenInnovation: 0,
                EconomyAndBusiness: 0,
                Work: 0,
                CommunityAndLife: 0,
                Patriotism: 0,
                totalVideos: 0
            };
        }

        stats[year].totalVideos++;

        // Map and count labels
        video.labels.forEach(labelStr => {
            const label = labelMap[labelStr];
            if (label) {
                switch (label) {
                    case FramingLabel.Environment: stats[year].Environment++; break;
                    case FramingLabel.GreenInnovation: stats[year].GreenInnovation++; break;
                    case FramingLabel.EconomyAndBusiness: stats[year].EconomyAndBusiness++; break;
                    case FramingLabel.Work: stats[year].Work++; break;
                    case FramingLabel.CommunityAndLife: stats[year].CommunityAndLife++; break;
                    case FramingLabel.Patriotism: stats[year].Patriotism++; break;
                }
            }
        });
    });

    return Object.values(stats).sort((a, b) => a.year - b.year);
};

// Get overall label distribution across all data
export const getOverallDistribution = (yearlyStats: YearlyStat[]) => {
    const totals = {
        Environment: 0,
        GreenInnovation: 0,
        EconomyAndBusiness: 0,
        Work: 0,
        CommunityAndLife: 0,
        Patriotism: 0,
        totalVideos: 0
    };

    yearlyStats.forEach(stat => {
        totals.Environment += stat.Environment;
        totals.GreenInnovation += stat.GreenInnovation;
        totals.EconomyAndBusiness += stat.EconomyAndBusiness;
        totals.Work += stat.Work;
        totals.CommunityAndLife += stat.CommunityAndLife;
        totals.Patriotism += stat.Patriotism;
        totals.totalVideos += stat.totalVideos;
    });

    return totals;
};

// Calculate percentage distributions per year
export const getYearlyPercentages = (yearlyStats: YearlyStat[]) => {
    return yearlyStats.map(stat => {
        const totalLabels = stat.Environment + stat.GreenInnovation +
            stat.EconomyAndBusiness + stat.Work +
            stat.CommunityAndLife + stat.Patriotism;

        return {
            year: stat.year,
            Environment: totalLabels > 0 ? (stat.Environment / totalLabels) * 100 : 0,
            GreenInnovation: totalLabels > 0 ? (stat.GreenInnovation / totalLabels) * 100 : 0,
            EconomyAndBusiness: totalLabels > 0 ? (stat.EconomyAndBusiness / totalLabels) * 100 : 0,
            Work: totalLabels > 0 ? (stat.Work / totalLabels) * 100 : 0,
            CommunityAndLife: totalLabels > 0 ? (stat.CommunityAndLife / totalLabels) * 100 : 0,
            Patriotism: totalLabels > 0 ? (stat.Patriotism / totalLabels) * 100 : 0,
            totalVideos: stat.totalVideos
        };
    });
};