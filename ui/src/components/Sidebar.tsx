
import React, { useState } from 'react';
import type { Model, FunctionDefinition } from '../lib/api';
import { Settings, Cpu, Activity, Wrench, History as HistoryIcon, MessageSquare } from 'lucide-react';

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
    onLoadModel: (alias: string) => void;
    onEjectModel: (alias: string) => void;
    isLoading: boolean;
    loadedModels: Record<string, string>; // alias -> model_id
    tools: FunctionDefinition[];
    history: { id: string, title: string, updated_at: string }[];
    currentConversationId: string | null;
    onSelectConversation: (id: string) => void;
    onNewChat: () => void;
    onDeleteConversation: (id: string) => void;
    loadingStatus: string | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
    models, selectedModelId, onSelectModel, config, onConfigChange, onLoadModel, onEjectModel, isLoading, loadedModels, tools,
    history, currentConversationId, onSelectConversation, onNewChat, onDeleteConversation, loadingStatus
}) => {
    // Single model mode
    const [deletingId, setDeletingId] = useState<string | null>(null);

    const handleDeleteClick = (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        setDeletingId(id);
    };

    const confirmDelete = (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        onDeleteConversation(id);
        setDeletingId(null);
    };

    const cancelDelete = (e: React.MouseEvent) => {
        e.stopPropagation();
        setDeletingId(null);
    };

    // Always check 'default' alias
    const isCurrentLoaded = loadedModels['default'] === selectedModelId;
    const currentLoadedId = loadedModels['default'];

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

                {/* History Section */}
                <div className="space-y-3">
                    <div className="flex justify-between items-center">
                        <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                            <HistoryIcon size={14} /> Recent
                        </label>
                        <button
                            onClick={onNewChat}
                            className="bg-white/5 hover:bg-white/10 text-xs px-2 py-1 rounded flex items-center gap-1 text-gray-300 transition-colors"
                        >
                            <MessageSquare size={12} /> New
                        </button>
                    </div>

                    <div className="space-y-1 max-h-40 overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-white/10">
                        {history.length === 0 && <div className="text-xs text-gray-600 italic px-2">No history</div>}
                        {history.map(h => (
                            <div
                                key={h.id}
                                onClick={() => onSelectConversation(h.id)}
                                className={`group w-full text-left text-xs px-3 py-2 rounded flex justify-between items-center cursor-pointer transition-colors ${currentConversationId === h.id
                                    ? 'bg-blue-500/20 text-blue-200 border border-blue-500/20'
                                    : 'text-gray-400 hover:bg-white/5 hover:text-gray-200'
                                    }`}
                                title={h.title}
                            >
                                <span className="truncate flex-1">{h.title}</span>
                                {deletingId === h.id ? (
                                    <div className="flex items-center gap-1 ml-2">
                                        <button
                                            onClick={(e) => confirmDelete(e, h.id)}
                                            className="text-red-400 hover:text-red-300 font-bold px-1"
                                        >
                                            ✓
                                        </button>
                                        <button
                                            onClick={cancelDelete}
                                            className="text-gray-500 hover:text-gray-400 px-1"
                                        >
                                            ✕
                                        </button>
                                    </div>
                                ) : (
                                    <button
                                        onClick={(e) => handleDeleteClick(e, h.id)}
                                        className="opacity-0 group-hover:opacity-100 p-1 text-gray-500 hover:text-red-400 transition-all"
                                        title="Delete"
                                    >
                                        ✕
                                    </button>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                <div className="border-t border-white/5 pt-4"></div>

                {/* Model Selection */}
                <div className="space-y-3">
                    <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-2 mb-2">
                        <Cpu size={14} /> Model Configuration
                    </label>

                    {/* Model Config Box */}
                    <div className="space-y-2 p-3 bg-black/10 rounded border border-white/5">
                        <div className="text-xs text-gray-400 mb-1 flex justify-between">
                            <span>Main Model</span>
                            {currentLoadedId && (
                                <span className="text-green-400 font-mono text-[10px] px-1 bg-green-900/20 rounded border border-green-900/30">
                                    ACTIVE
                                </span>
                            )}
                        </div>

                        {isLoading && loadingStatus && (
                            <div className="text-[10px] text-blue-400 font-mono animate-pulse mb-1">
                                &gt; {loadingStatus}
                            </div>
                        )}

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
                        <div className="flex gap-2">
                            <button
                                onClick={() => onLoadModel('default')}
                                disabled={isLoading || !selectedModelId || isCurrentLoaded}
                                className={`flex-1 text-xs font-bold py-2 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed bg-blue-600 hover:bg-blue-500 text-white`}
                            >
                                {isLoading ? 'LOADING...' : (isCurrentLoaded ? 'LOADED' : 'LOAD')}
                            </button>
                            {currentLoadedId && (
                                <button
                                    onClick={() => onEjectModel('default')}
                                    disabled={isLoading}
                                    className="px-3 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 text-red-500 text-xs font-bold rounded transition-colors"
                                    title="Eject Model"
                                >
                                    ✕
                                </button>
                            )}
                        </div>
                        {currentLoadedId && (
                            <div className="text-[10px] text-gray-500 truncate mt-1">
                                Loaded: {currentLoadedId}
                            </div>
                        )}
                    </div>
                </div>

                {/* Parameters (Only for Agent strictly speaking, but keep global for now) */}
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
                v1.1.0 • Localhost
            </div>
        </div>
    );
};
