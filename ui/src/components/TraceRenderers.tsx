
import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Terminal, Brain, Zap, Link as LinkIcon } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface Block {
    type: 'thought' | 'action_choice' | 'function_call' | 'output' | 'text' | 'final_answer';
    content: string;
    id?: string;
}

const parseContent = (text: string): Block[] => {
    // Simple heuristic parsing based on IRCA dataset format
    const blocks: Block[] = [];
    const lines = text.split('\n');

    let currentBuffer = '';
    let currentType: Block['type'] = 'text';
    let currentId: string | undefined;

    const flush = () => {
        if (currentBuffer.trim()) {
            blocks.push({
                type: currentType,
                content: currentBuffer.trim(),
                id: currentId
            });
        }
        currentBuffer = '';
        currentId = undefined;
    };

    for (const line of lines) {
        if (line.startsWith('Thought:')) {
            flush();
            currentType = 'thought';
            currentBuffer = line.replace('Thought:', '').trim();
        } else if (line.startsWith('Action choice:')) {
            flush();
            blocks.push({ type: 'action_choice', content: line.replace('Action choice:', '').trim() });
            currentType = 'text'; // Reset after single line
        } else if (line.startsWith('Call function:')) {
            flush();
            currentBuffer = line.replace('Call function:', '').trim();
            currentType = 'function_call';
        } else if (line.startsWith('Output[')) {
            flush();
            // Extract content and ID: Output[ID]: content
            // Regex to match Output[ID] or Output[ID]...:
            const match = line.match(/^Output\[(.*?)\](?::\s*)?(.*)/);
            if (match) {
                currentId = match[1]; // The ID part inside brackets
                let content = match[2]; // Content after 'Output[ID]: '

                // If regex captured content, use it, otherwise use remainder of line or nothing
                // The 'match' logic above assumes standard formatting
                if (!content && !line.includes(':')) {
                    // Maybe just Output[ID] with no colon? Unlikely given prompt constants.
                    content = line.replace(/^Output\[.*?\]/, '').trim();
                }
                blocks.push({ type: 'output', content: content.trim(), id: currentId });
            } else {
                blocks.push({ type: 'output', content: line });
            }
            currentType = 'text';
        } else if (line.startsWith('### FINAL ANSWER')) {
            flush();
            currentType = 'final_answer';
        } else if (line.trim() === '<|wait|>') {
            // Skip wait token
            continue;
        } else {
            currentBuffer += (currentBuffer ? '\n' : '') + line;
        }
    }
    flush();

    return blocks;
};

export const ThoughtBlock: React.FC<{ content: string }> = ({ content }) => {
    const [isOpen, setIsOpen] = useState(false);
    return (
        <div className="my-2 border border-gray-700/50 rounded-lg overflow-hidden bg-gray-900/30">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center gap-2 px-3 py-2 text-xs font-medium text-gray-400 hover:bg-gray-800/50 transition-colors"
            >
                <Brain size={14} className="text-purple-400" />
                <span className="uppercase tracking-wider">Reasoning Process</span>
                <div className="flex-1" />
                {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </button>
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="border-t border-gray-700/50"
                    >
                        <div className="p-3 text-sm text-gray-300 font-mono leading-relaxed whitespace-pre-wrap">
                            {content}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export const FunctionCallBlock: React.FC<{ content: string }> = ({ content }) => {
    let formatted = content;
    try {
        formatted = JSON.stringify(JSON.parse(content), null, 2);
    } catch (e) { }

    return (
        <div className="my-2 font-mono text-xs">
            <div className="flex items-center gap-2 text-blue-400 mb-1">
                <Terminal size={12} />
                <span className="font-bold uppercase">Function Call</span>
            </div>
            <div className="bg-black/50 border border-blue-900/30 rounded p-3 text-blue-300 overflow-x-auto">
                <pre>{formatted}</pre>
            </div>
        </div>
    );
};

export const OutputBlock: React.FC<{ content: string, id?: string }> = ({ content, id }) => (
    <div
        id={id ? `output-${id}` : undefined}
        className="my-2 pl-3 border-l-2 border-green-500/50 transition-all duration-300"
    >
        <div className="text-xs text-green-500/70 mb-1 uppercase font-bold flex items-center gap-1">
            <Zap size={10} /> Tool Output {id && <span className="opacity-50 text-[10px] font-normal font-mono">#{id}</span>}
        </div>
        <div className="text-sm text-gray-300 font-mono transition-colors">{content}</div>
    </div>
);

// Component to handle markdown-like links [label](Output[ID])
const ReferenceLink: React.FC<{ label: string, outputId: string }> = ({ label, outputId }) => {
    const handleMouseEnter = () => {
        const el = document.getElementById(`output-${outputId}`);
        if (el) {
            el.classList.add('bg-green-500/10', 'pl-4'); // Highlight effect
            el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    };

    const handleMouseLeave = () => {
        const el = document.getElementById(`output-${outputId}`);
        if (el) {
            el.classList.remove('bg-green-500/10', 'pl-4');
        }
    };

    return (
        <span
            className="inline-flex items-center gap-0.5 text-blue-400 hover:text-blue-300 cursor-pointer bg-blue-500/10 hover:bg-blue-500/20 px-1 rounded transition-colors"
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
        >
            <LinkIcon size={10} />
            {label}
        </span>
    );
};

const RichFinalAnswer: React.FC<{ content: string }> = ({ content }) => {
    // Regex for [label](Output[ID])
    // Split the content by the regex to interleave text and components
    const regex = /(\[.*?\]\(Output\[.*?\]\))/g;
    const parts = content.split(regex);

    return (
        <div className="text-md text-gray-100 whitespace-pre-wrap leading-relaxed">
            {parts.map((part, i) => {
                const match = part.match(/^\[(.*?)\]\(Output\[(.*?)\]\)$/);
                if (match) {
                    return <ReferenceLink key={i} label={match[1]} outputId={match[2]} />;
                }
                return <span key={i}>{part}</span>;
            })}
        </div>
    );
};

export const Renderer: React.FC<{ text: string }> = ({ text }) => {
    const blocks = parseContent(text);

    return (
        <div className="space-y-1">
            {blocks.map((block, i) => {
                switch (block.type) {
                    case 'thought':
                        return <ThoughtBlock key={i} content={block.content} />;
                    case 'function_call':
                        return <FunctionCallBlock key={i} content={block.content} />;
                    case 'output':
                        return <OutputBlock key={i} content={block.content} id={block.id} />;
                    case 'action_choice':
                        return null; // Don't render action choice explicitly
                    case 'final_answer':
                        return (
                            <div key={i} className="mt-4 pt-4 border-t border-gray-700">
                                <div className="text-xs text-gray-500 mb-2 uppercase tracking-widest">Final Answer</div>
                                <RichFinalAnswer content={block.content} />
                            </div>
                        );
                    default:
                        return <div key={i} className="whitespace-pre-wrap text-gray-200">{block.content}</div>;
                }
            })}
        </div>
    );
};
