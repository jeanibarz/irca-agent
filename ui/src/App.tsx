
import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatInterface } from './components/ChatInterface';
import { api } from './lib/api';
import type { Message, Model } from './lib/api';

function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const [config, setConfig] = useState({
    temperature: 0.7,
    maxTokens: 4096,
    topP: 1.0
  });

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const list = await api.listModels();
      setModels(list);
      if (list.length > 0) setSelectedModelId(list[0].path);
    } catch (e) {
      console.error("Failed to list models", e);
    }
  };

  const handleLoadModel = async () => {
    if (!selectedModelId) return;
    setIsLoading(true);
    try {
      // Detect if it is an adapter or base based on type
      const model = models.find(m => m.path === selectedModelId);
      if (model?.type === 'adapter') {
        // Assuming base is mistral-v3 for now for adapters, or we need metadata
        // For RFC Phase 1 simplicity, hardcode base or infer
        await api.loadModel("mistralai/Mistral-7B-Instruct-v0.3", selectedModelId);
      } else {
        await api.loadModel(selectedModelId);
      }
    } catch (e) {
      console.error("Failed to load model", e);
      alert("Failed to load model check console");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (content: string) => {
    const userMsg: Message = { role: 'user', content };
    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setIsGenerating(true);

    try {
      // Dummy functions for demo "Wow" factor
      const dummyFunctions = [
        {
          name: "get_weather",
          description: "Get current weather",
          parameters: {
            type: "object",
            properties: {
              location: { type: "string" },
              unit: { type: "string", enum: ["c", "f"] }
            },
            required: ["location"]
          }
        }
      ];

      const response = await api.generate({
        messages: newHistory,
        functions: dummyFunctions,
        max_tokens: config.maxTokens,
        temperature: config.temperature,
        top_p: config.topP
      });

      setMessages([...newHistory, { role: 'assistant', content: response.content }]);
    } catch (e) {
      console.error("Generation failed", e);
      setMessages([...newHistory, { role: 'assistant', content: "Error: Generation failed. Is the model loaded?" }]);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-[#121212] text-white overflow-hidden">
      <Sidebar
        models={models}
        selectedModelId={selectedModelId}
        onSelectModel={setSelectedModelId}
        config={config}
        onConfigChange={(k, v) => setConfig(prev => ({ ...prev, [k]: v }))}
        onLoadModel={handleLoadModel}
        isLoading={isLoading}
      />
      <main className="flex-1 h-full min-w-0">
        <ChatInterface
          messages={messages}
          onSendMessage={handleSendMessage}
          isGenerating={isGenerating}
        />
      </main>
    </div>
  );
}

export default App;
