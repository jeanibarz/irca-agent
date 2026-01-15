
import React, { useState, useEffect, useRef } from 'react';
import { Send, TerminalSquare, User, Sparkles, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import { Renderer } from './TraceRenderers';
import { api } from '../lib/api';
import type { FunctionDefinition } from '../lib/api';

interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string;
}

interface ChatInterfaceProps {
    messages: Message[];
    onSendMessage: (content: string) => void;
    isGenerating: boolean;
    tools: FunctionDefinition[];
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ messages, onSendMessage, isGenerating, tools }) => {
    const [input, setInput] = useState('');
    const [isSynthesizing, setIsSynthesizing] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = () => {
        if (!input.trim() || isGenerating) return;
        onSendMessage(input);
        setInput('');
    };

    const handleSynthetic = async (type: 'feasible' | 'infeasible') => {
        setIsSynthesizing(true);
        try {
            const { query } = await api.generateSyntheticQuery(tools, type);
            // Typewriter effect
            setInput('');
            let i = 0;
            const typing = setInterval(() => {
                setInput(query.slice(0, i + 1));
                i++;
                if (i >= query.length) clearInterval(typing);
            }, 30);
        } catch (e) {
            console.error(e);
            alert("Failed to generate synthetic query. Ensure a model is loaded.");
        } finally {
            setIsSynthesizing(false);
        }
    };

    return (
        <div className="flex-1 flex flex-col h-full relative bg-[#121212]">
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6">
                {messages.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center text-gray-600 opacity-50">
                        <TerminalSquare size={48} className="mb-4" />
                        <p>No messages yet. Start the conversation.</p>
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <motion.div
                        key={idx}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                        {msg.role === 'assistant' && (
                            <div className="w-8 h-8 rounded bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shrink-0 shadow-lg shadow-purple-900/20">
                                <TerminalSquare size={16} className="text-white" />
                            </div>
                        )}

                        <div className={`max-w-[85%] ${msg.role === 'user' ? 'bg-blue-600/20 border border-blue-500/30' : 'bg-[#1e1e1e] border border-white/5'} rounded-2xl p-4 shadow-sm`}>
                            {msg.role === 'user' ? (
                                <div className="text-sm text-blue-100">{msg.content}</div>
                            ) : (
                                <Renderer text={msg.content} />
                            )}
                        </div>

                        {msg.role === 'user' && (
                            <div className="w-8 h-8 rounded bg-gray-700 flex items-center justify-center shrink-0">
                                <User size={16} className="text-gray-300" />
                            </div>
                        )}
                    </motion.div>
                ))}

                {isGenerating && (
                    <div className="flex gap-4">
                        <div className="w-8 h-8 rounded bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shrink-0 animate-pulse">
                            <TerminalSquare size={16} className="text-white" />
                        </div>
                        <div className="text-xs text-gray-500 flex items-center h-8">Generating...</div>
                    </div>
                )}
                <div ref={bottomRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 bg-[#121212] border-t border-white/5">
                <div className="max-w-4xl mx-auto space-y-3">
                    {/* Action Buttons */}
                    <div className="flex justify-center gap-3">
                        <button
                            onClick={() => handleSynthetic('feasible')}
                            disabled={isSynthesizing || isGenerating}
                            className="flex items-center gap-2 px-3 py-1.5 bg-green-500/10 hover:bg-green-500/20 border border-green-500/20 rounded-lg text-green-400 text-xs transition-colors disabled:opacity-50"
                        >
                            <Sparkles size={12} />
                            Generate Feasible Query
                        </button>
                        <button
                            onClick={() => handleSynthetic('infeasible')}
                            disabled={isSynthesizing || isGenerating}
                            className="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-lg text-red-400 text-xs transition-colors disabled:opacity-50"
                        >
                            <AlertCircle size={12} />
                            Generate Infeasible Query
                        </button>
                    </div>

                    <div className="relative group">
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSend();
                                }
                            }}
                            placeholder="Type a message or use the generators above..."
                            className="w-full bg-[#1e1e1e] border border-white/10 rounded-xl px-4 py-3 pr-12 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/20 transition-all resize-none shadow-xl"
                            rows={1}
                        />
                        <button
                            onClick={handleSend}
                            disabled={!input.trim() || isGenerating}
                            className="absolute right-2 top-2 p-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-0 disabled:translate-x-2 text-white rounded-lg transition-all duration-200 shadow-lg shadow-blue-900/20"
                        >
                            <Send size={16} />
                        </button>
                        <div className="absolute -top-6 right-0 text-[10px] text-gray-600 opacity-0 group-hover:opacity-100 transition-opacity">
                            Press Enter to send, Shift+Enter for new line
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
