
import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatInterface } from './components/ChatInterface';
import { api } from './lib/api';
import type { Message, Model, FunctionDefinition } from './lib/api';

const DUMMY_TOOLS: FunctionDefinition[] = [
  {
    name: "get_weather",
    description: "Get the current weather conditions for a specific location.",
    parameters: {
      type: "object",
      properties: {
        location: { type: "string", description: "The city and state, e.g. San Francisco, CA" },
        unit: { type: "string", enum: ["celsius", "fahrenheit"] }
      },
      required: ["location"]
    }
  },
  {
    name: "get_stock_price",
    description: "Retrieve the current stock price for a given ticker symbol.",
    parameters: {
      type: "object",
      properties: {
        ticker: { type: "string", description: "The stock ticker symbol, e.g. AAPL" }
      },
      required: ["ticker"]
    }
  },
  {
    name: "calculator",
    description: "Perform basic mathematical operations.",
    parameters: {
      type: "object",
      properties: {
        expression: { type: "string", description: "The mathematical expression to evaluate, e.g. '2 + 2'" }
      },
      required: ["expression"]
    }
  }
];

function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [loadedModelId, setLoadedModelId] = useState<string | null>(null);

  // History State
  const [history, setHistory] = useState<{ id: string, title: string, updated_at: string }[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);

  const [config, setConfig] = useState({
    temperature: 0.7,
    maxTokens: 4096,
    topP: 1.0
  });

  useEffect(() => {
    loadModels();
    loadHistory();
    fetchCurrentModel();
    // Poll every 3s to keep state in sync
    const interval = setInterval(fetchCurrentModel, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchCurrentModel = async () => {
    try {
      const current = await api.getCurrentModel();
      setLoadedModelId(current.adapter_id || current.base_model_id);
    } catch (e) {
      // quiet fail on poll
    }
  };

  const loadModels = async () => {
    try {
      const list = await api.listModels();
      setModels(list);
      if (list.length > 0) setSelectedModelId(list[0].path);
    } catch (e) {
      console.error("Failed to list models", e);
    }
  };

  const loadHistory = async () => {
    try {
      const list = await api.listConversations();
      setHistory(list);
    } catch (e) {
      console.error("Failed to load history", e);
    }
  };

  const handleNewChat = async () => {
    try {
      const conv = await api.createConversation();
      setMessages([]);
      setCurrentConversationId(conv.id);
      await loadHistory();
    } catch (e) {
      console.error("Failed to create chat", e);
    }
  };

  const handleSelectConversation = async (id: string) => {
    try {
      const conv = await api.getConversation(id);
      setMessages(conv.messages);
      setCurrentConversationId(conv.id);
    } catch (e) {
      console.error("Failed to load conversation", e);
    }
  };

  const handleLoadModel = async () => {
    if (!selectedModelId) return;
    setIsLoading(true);
    try {
      // Detect if it is an adapter or base based on type
      const model = models.find(m => m.path === selectedModelId);
      if (model?.type === 'adapter') {
        const base = "mistralai/Mistral-7B-Instruct-v0.3";
        await api.loadModel(base, selectedModelId);
      } else {
        await api.loadModel(selectedModelId);
      }
      await fetchCurrentModel();
    } catch (e: any) {
      console.error("Failed to load model", e);
      if (e.response && e.response.status === 409) {
        alert("Model loading already in progress!");
      } else {
        alert("Failed to load model. Check console.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleEjectModel = async () => {
    if (!confirm("Are you sure you want to eject the model? This will clear GPU memory.")) return;
    try {
      await api.ejectModel();
      setLoadedModelId(null);
    } catch (e) {
      console.error("Failed to eject", e);
    }
  };

  const handleSendMessage = async (content: string) => {
    let distinctId = currentConversationId;

    // Auto-create chat if none exists
    if (!distinctId) {
      try {
        const conv = await api.createConversation();
        distinctId = conv.id;
        setCurrentConversationId(distinctId);
      } catch (e) {
        console.error("Failed to create implicit chat", e);
        alert("Failed to start conversation.");
        return;
      }
    }

    const userMsg: Message = { role: 'user', content };
    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setIsGenerating(true);

    // Save interim
    if (distinctId) {
      api.updateConversation(distinctId, newHistory).catch(console.error);
    }

    try {
      const response = await api.generate({
        messages: newHistory,
        functions: DUMMY_TOOLS,
        max_tokens: config.maxTokens,
        temperature: config.temperature,
        top_p: config.topP
      });

      // FIX: Trim <|wait|> from response content before displaying
      const cleanContent = response.content.replace(/<\|wait\|>/g, '').trim();

      const assistantMsg: Message = { role: 'assistant', content: cleanContent };
      const finalHistory = [...newHistory, assistantMsg];
      setMessages(finalHistory);

      if (distinctId) {
        await api.updateConversation(distinctId, finalHistory);
        loadHistory(); // Update titles
      }
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
        onEjectModel={handleEjectModel}
        isLoading={isLoading}
        loadedModelId={loadedModelId}
        tools={DUMMY_TOOLS}
        history={history}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
      />
      <main className="flex-1 h-full min-w-0">
        <ChatInterface
          messages={messages}
          onSendMessage={handleSendMessage}
          isGenerating={isGenerating}
          tools={DUMMY_TOOLS}
        />
      </main>
    </div>
  );
}

export default App;
