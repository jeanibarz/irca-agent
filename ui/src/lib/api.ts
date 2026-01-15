
import axios from 'axios';

const API_BASE = 'http://localhost:8000/v1';

export interface Model {
    id: string;
    type: 'base' | 'adapter';
    path: string;
}

export interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string;
}

export interface FunctionDefinition {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
}

export interface GenerationRequest {
    messages: Message[];
    functions?: FunctionDefinition[];
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
    model_id?: string;
    adapter_id?: string;
}

export interface GenerationResponse {
    role: string;
    content: string;
    finish_reason: string;
}

export const api = {
    listModels: async (): Promise<Model[]> => {
        const res = await axios.get(`${API_BASE}/models`);
        return res.data;
    },

    loadModel: async (baseModelId: string, adapterId?: string): Promise<{ status: string, message: string }> => {
        const res = await axios.post(`${API_BASE}/model/load`, { base_model_id: baseModelId, adapter_id: adapterId });
        return res.data;
    },

    generate: async (req: GenerationRequest): Promise<GenerationResponse> => {
        const res = await axios.post(`${API_BASE}/chat/completions`, req);
        return res.data;
    }
};
