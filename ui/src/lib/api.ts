
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

    loadModel: async (baseModelId: string, adapterId?: string, alias: string = "default"): Promise<{ status: string, message: string }> => {
        const res = await axios.post(`${API_BASE}/model/load`, { base_model_id: baseModelId, adapter_id: adapterId, alias });
        return res.data;
    },

    generate: async (req: GenerationRequest): Promise<GenerationResponse> => {
        const res = await axios.post(`${API_BASE}/chat/completions`, req);
        return res.data;
    },

    generateSyntheticQuery: async (tools: FunctionDefinition[], type: 'feasible' | 'infeasible'): Promise<{ query: string }> => {
        const res = await axios.post(`${API_BASE}/synthetic/query`, { tools, type });
        return res.data;
    },

    getCurrentModel: async (): Promise<Record<string, { base: string, adapter: string | null }>> => {
        const res = await axios.get(`${API_BASE}/model/current`);
        return res.data;
    },

    ejectModel: async (alias: string = "default"): Promise<{ status: string, message: string }> => {
        const res = await axios.post(`${API_BASE}/model/eject`, { alias });
        return res.data;
    },

    listConversations: async (): Promise<{ id: string, title: string, updated_at: string }[]> => {
        const res = await axios.get(`${API_BASE}/conversations`);
        return res.data;
    },

    createConversation: async (): Promise<{ id: string, title: string }> => {
        const res = await axios.post(`${API_BASE}/conversations`, { title: "New Chat" });
        return res.data;
    },

    getConversation: async (id: string): Promise<{ id: string, title: string, messages: Message[] }> => {
        const res = await axios.get(`${API_BASE}/conversations/${id}`);
        return res.data;
    },

    updateConversation: async (id: string, messages: Message[]): Promise<void> => {
        await axios.post(`${API_BASE}/conversations/${id}`, { messages });
    },

    deleteConversation: async (id: string): Promise<void> => {
        await axios.delete(`${API_BASE}/conversations/${id}`);
    }
};
