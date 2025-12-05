export enum FramingLabel {
  CommunityAndLife = "Community and Life",
  EconomyAndBusiness = "Economy and Business",
  Work = "Work",
  Environment = "Environment",
  GreenInnovation = "Green Innovation",
  Patriotism = "Patriotism"
}

export interface YearlyStat {
  year: number;
  Environment: number;
  GreenInnovation: number;
  EconomyAndBusiness: number;
  Work: number;
  CommunityAndLife: number;
  Patriotism: number;
  totalVideos: number;
}

// Restricted to user request
export const LABELS = [
  FramingLabel.CommunityAndLife,
  FramingLabel.Work,
  FramingLabel.Environment,
];

export const COLORS: Record<FramingLabel, string> = {
  [FramingLabel.CommunityAndLife]: "#f59e0b", // Amber
  [FramingLabel.EconomyAndBusiness]: "#94a3b8", // Slate (Inactive)
  [FramingLabel.Work]: "#3b82f6", // Blue (Now prominent)
  [FramingLabel.Environment]: "#10b981", // Emerald
  [FramingLabel.GreenInnovation]: "#8b5cf6", // Violet
  [FramingLabel.Patriotism]: "#ef4444", // Red
};