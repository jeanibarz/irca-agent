import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import get_settings
from server.model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()


class ModelInfo(BaseModel):
    id: str
    type: str  # "base" or "adapter"
    path: str


class LoadModelRequest(BaseModel):
    base_model_id: str
    adapter_id: str | None = None


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """
    List available models and adapters.
    """
    models = []

    # 1. Base Models (Hardcoded presets)
    presets = ["mistral", "mistral-v3", "tinyllama"]
    for preset in presets:
        config = settings.get_model_config(preset)
        models.append(ModelInfo(id=config["base_model"], type="base", path=config["base_model"]))

    # 2. Adapters (Scan directory)
    finetuned_dir = settings.finetuned_models_path
    logger.info(f"Scanning for adapters in: {finetuned_dir}")
    logger.info(f"Directory exists: {finetuned_dir.exists()}")
    if finetuned_dir.exists():
        for item in finetuned_dir.iterdir():
            logger.info(f"Found item: {item.name}, is_dir: {item.is_dir()}")
            if item.is_dir():
                models.append(ModelInfo(id=item.name, type="adapter", path=str(item)))

    logger.info(f"Total models found: {len(models)}")
    return models


@router.post("/model/load")
async def load_model(request: LoadModelRequest) -> dict[str, str]:
    """
    Load a model into memory.
    """
    manager = ModelManager.get_instance()
    try:
        # Check if adapter_id is a short name or path
        adapter_path = None
        if request.adapter_id:
            finetuned_dir = settings.finetuned_models_path
            potential_path = finetuned_dir / request.adapter_id
            if potential_path.exists():
                adapter_path = str(potential_path)
            else:
                adapter_path = request.adapter_id  # Assume absolute or HF ID

        await manager.load_model(request.base_model_id, adapter_path)
        return {"status": "success", "message": f"Loaded {request.base_model_id} with {adapter_path}"}
    except RuntimeError as e:
        # Busy
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/model/current")
async def get_current_model() -> dict[str, str | None]:
    manager = ModelManager.get_instance()
    return {"base_model_id": manager.current_base_model_id, "adapter_id": manager.current_adapter_id}


@router.post("/model/eject")
async def eject_model() -> dict[str, str]:
    manager = ModelManager.get_instance()
    try:
        await manager.eject_model()
        return {"status": "success", "message": "Model ejected"}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
