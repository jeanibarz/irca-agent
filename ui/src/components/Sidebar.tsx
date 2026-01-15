
import React from 'react';
import type { Model } from '../lib/api';
import { Settings, Cpu, Activity } from 'lucide-react';

interface SidebarProps {
    models: Model[];
    selectedModelId: string;
    onSelectModel: (id: string) => void;
    config: {
        temperature: number;
        maxTokens: number;
        topP: number;
    };
    onConfigChange: (key: string, value: number) => void;
    onLoadModel: () => void;
    isLoading: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
    models, selectedModelId, onSelectModel, config, onConfigChange, onLoadModel, isLoading
}) => {
    return (
        <div className="w-80 h-full bg-[#1e1e1e] border-r border-white/5 flex flex-col">
            <div className="p-4 border-b border-white/5">
                <h1 className="text-lg font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent flex items-center gap-2">
                    <Activity className="text-blue-400" size={20} />
                    IRCA Agent
                </h1>
                <p className="text-xs text-gray-500 mt-1">Playground & Eval Suite</p>
            </div>

            <div className="p-4 space-y-6 flex-1 overflow-y-auto">
                {/* Model Selection */}
                <div className="space-y-3">
                    <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                        <Cpu size={14} /> Model
                    </label>
                    <div className="space-y-2">
                        <select
                            value={selectedModelId}
                            onChange={(e) => onSelectModel(e.target.value)}
                            className="w-full bg-[#2a2a2a] border border-white/10 rounded px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500/50 [&>option]:bg-[#2a2a2a] [&>option]:text-gray-200"
                        >
                            <option value="" disabled>Select a model</option>
                            {models.map(m => (
                                <option key={m.path} value={m.path}>
                                    {m.id} ({m.type})
                                </option>
                            ))}
                        </select>
                        <button
                            onClick={onLoadModel}
                            disabled={isLoading || !selectedModelId}
                            className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-bold py-2 rounded transition-colors"
                        >
                            {isLoading ? 'LOADING...' : 'LOAD MODEL TO GPU'}
                        </button>
                    </div>
                </div>

                {/* Parameters */}
                <div className="space-y-4">
                    <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                        <Settings size={14} /> Hyperparameters
                    </label>

                    {/* Temp */}
                    <div className="space-y-1">
                        <div className="flex justify-between text-xs text-gray-500">
                            <span>Temperature</span>
                            <span>{config.temperature}</span>
                        </div>
                        <input
                            type="range" min="0" max="2" step="0.1"
                            value={config.temperature}
                            onChange={(e) => onConfigChange('temperature', parseFloat(e.target.value))}
                            className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                        />
                    </div>

                    {/* Top P */}
                    <div className="space-y-1">
                        <div className="flex justify-between text-xs text-gray-500">
                            <span>Top P</span>
                            <span>{config.topP}</span>
                        </div>
                        <input
                            type="range" min="0" max="1" step="0.05"
                            value={config.topP}
                            onChange={(e) => onConfigChange('topP', parseFloat(e.target.value))}
                            className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                        />
                    </div>

                    {/* Max Tokens */}
                    <div className="space-y-1">
                        <div className="flex justify-between text-xs text-gray-500">
                            <span>Max Tokens</span>
                            <span>{config.maxTokens}</span>
                        </div>
                        <input
                            type="range" min="256" max="8192" step="256"
                            value={config.maxTokens}
                            onChange={(e) => onConfigChange('maxTokens', parseInt(e.target.value))}
                            className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                        />
                    </div>
                </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-white/5 text-xs text-gray-600 text-center">
                v1.0.0 • Localhost
            </div>
        </div>
    );
};
