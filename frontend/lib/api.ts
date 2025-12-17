import axios from 'axios';

// Smart API URL configuration:
// - Production (Vercel): Use NEXT_PUBLIC_API_URL from environment
// - Development: Use empty string to leverage Next.js proxy
const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

// Create an axios instance with default config
const apiClient = axios.create({
  baseURL: API_URL,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Session & Basic Info
  getSession: async () => {
    try {
      const res = await apiClient.get('/api/session');
      return res.data;
    } catch (error) {
      console.error("Error fetching session:", error);
      throw error;
    }
  },

  uploadFile: async (file: File, options?: any) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      if (options) {
        Object.keys(options).forEach(key => {
          formData.append(key, String(options[key]));
        });
      }

      const res = await apiClient.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return res.data;
    } catch (error) {
      console.error("Error uploading file:", error);
      throw error;
    }
  },

  // Analysis & Overview
  analyzeData: async () => {
    const res = await apiClient.post('/api/analyze');
    return res.data;
  },

  getInsights: async () => {
    const res = await apiClient.get('/api/insights');
    return res.data;
  },

  // Visualization
  getRecommendations: async () => {
    const res = await apiClient.get('/api/recommendations');
    return res.data;
  },

  createVisualization: async (type: string, config: any) => {
    const res = await apiClient.post('/api/visualize', { type, config });
    return res.data;
  },

  // Advanced: Explorer & Columns
  getColumnData: async (column: string) => {
    const res = await apiClient.get(`/api/column/${column}`);
    return res.data;
  },

  filterData: async (filters: any) => {
    const res = await apiClient.post('/api/filter', { filters });
    return res.data;
  },

  // Advanced: Insights Chat
  chat: async (question: string) => {
    const res = await apiClient.post('/api/chat', { question });
    return res.data;
  },

  // Custom Viz
  createCustomViz: async (chartType: string, config: any) => {
    // chart_type matches backend expectation
    const res = await apiClient.post('/api/chart/custom', { chart_type: chartType, config });
    return res.data;
  },

  exportData: async (filters: any) => {
    const res = await apiClient.post('/api/export', { filters }, {
      responseType: 'blob', // Important for file handling
    });
    return res.data;
  }
};
