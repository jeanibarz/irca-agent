
import React from 'react';
import type { Model, FunctionDefinition } from '../lib/api';
import { Settings, Cpu, Activity, Wrench } from 'lucide-react';

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
    onEjectModel: () => void;
    isLoading: boolean;
    loadedModelId: string | null;
    tools: FunctionDefinition[];
}

export const Sidebar: React.FC<SidebarProps> = ({
    models, selectedModelId, onSelectModel, config, onConfigChange, onLoadModel, onEjectModel, isLoading, loadedModelId, tools
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
                                    {loadedModelId === m.path ? ' [LOADED]' : ''}
                                </option>
                            ))}
                        </select>
                        <div className="flex gap-2">
                            <button
                                onClick={onLoadModel}
                                disabled={isLoading || !selectedModelId || loadedModelId === selectedModelId}
                                className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-bold py-2 rounded transition-colors"
                            >
                                {isLoading ? 'LOADING...' : (loadedModelId === selectedModelId ? 'LOADED' : 'LOAD')}
                            </button>
                            {loadedModelId && (
                                <button
                                    onClick={onEjectModel}
                                    disabled={isLoading}
                                    className="px-3 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 text-red-500 text-xs font-bold rounded transition-colors"
                                    title="Eject Model"
                                >
                                    ✕
                                </button>
                            )}
                        </div>
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

                {/* Available Tools */}
                <div className="space-y-4">
                    <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                        <Wrench size={14} /> Available Tools
                    </label>
                    <div className="space-y-2">
                        {tools.map((tool) => (
                            <div key={tool.name} className="bg-black/20 border border-white/5 rounded p-3">
                                <div className="text-sm font-mono text-blue-300 font-bold mb-1">{tool.name}</div>
                                <div className="text-xs text-gray-400 mb-2 leading-relaxed">{tool.description}</div>
                                <div className="flex flex-wrap gap-1">
                                    {Object.keys(tool.parameters.properties || {}).map(f => (
                                        <span key={f} className="text-[10px] text-gray-400 font-mono bg-white/5 px-1.5 py-0.5 rounded border border-white/5">
                                            {f}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        ))}
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
