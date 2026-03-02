import axios from 'axios';

const API_URL = ''; // Relative path for self-hosted

export const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const predictStress = async (data) => {
    const response = await api.post('/api/predict-stress', data);
    return response.data;
};

export const submitTest = async (data) => {
    const response = await api.post('/api/submit-test', data);
    return response.data;
};

export const getTestResults = async (id, token) => {
    const response = await api.get(`/api/test-results/${id}`, {
        params: { token }
    });
    return response.data;
};
