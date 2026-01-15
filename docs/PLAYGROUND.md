# IRCA Playground & Evaluation Suite

## Quick Start

### Backend (FastAPI)
```bash
# Terminal 1: Start the backend server
WORKSPACE_DIR=/home/jean/git/irca-agent poetry run uvicorn src.server.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (React/Vite)
```bash
# Terminal 2: Start the frontend dev server
cd ui
npm run dev -- --host 0.0.0.0 --port 3000
```

### Access
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Usage

1. **Select a Model**: Choose from the dropdown (base models or finetuned adapters)
2. **Load to GPU**: Click "LOAD MODEL TO GPU" button (takes ~30s for first load)
3. **Configure Parameters**: Adjust Temperature, Top-P, and Max Tokens
4. **Chat**: Type your message and press Enter

## Features

### Current (Phase 1 & 2 - Implemented)
- ✅ **Backend API**: FastAPI with model management endpoints
- ✅ **Stateful Model Loading**: Keep model in GPU memory across requests
- ✅ **Model Switching**: Support for base models and LoRA adapters
- ✅ **Premium UI**: Dark mode with glassmorphism and smooth animations
- ✅ **Rich Trace Rendering**: Specialized display for Thoughts, Function Calls, and Outputs
- ✅ **Hyperparameter Control**: Real-time adjustment of generation params
- ✅ **Max Tokens**: Default 4096 (configurable up to 8192)

### Planned (Phase 3 - Judge Integration)
- ⏳ **AI Judge**: LLM-based evaluation of trace quality
- ⏳ **Scoring System**: Automated metrics for function calling accuracy
- ⏳ **Function Sandbox**: Real execution of function calls (currently dummy)

## Architecture

```
┌─────────────┐      HTTP      ┌─────────────┐
│   React UI  │ ────────────► │  FastAPI    │
│  (Port 3000)│                │  (Port 8000)│
└─────────────┘                └──────┬──────┘
                                      │
                                      ▼
                                ┌─────────────┐
                                │ Model       │
                                │ Manager     │
                                │ (GPU Cache) │
                                └─────────────┘
```

## API Endpoints

### Models
- `GET /v1/models` - List available models
- `POST /v1/model/load` - Load model into GPU memory

### Generation
- `POST /v1/chat/completions` - Generate completion (OpenAI-compatible format)

## Development Notes

- **Model Path Resolution**: Adapters are auto-discovered from `models/finetuned_models/`
- **CORS**: Enabled for localhost development
- **Hot Reload**: Both frontend and backend support live reloading
