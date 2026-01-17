import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import get_settings
from src.server.model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()


def get_adapter_base_model(adapter_path: str) -> str | None:
    """Read base model from adapter_config.json."""
    config_path = Path(adapter_path) / "adapter_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            return config.get("base_model_name_or_path")
        except (json.JSONDecodeError, OSError):
            return None
    return None


class ModelInfo(BaseModel):
    id: str
    type: str  # "base" or "adapter"
    path: str
    base_model: str | None = None  # For adapters, the required base model


class LoadModelRequest(BaseModel):
    base_model_id: str
    adapter_id: str | None = None
    alias: str = "default"  # e.g., "default", "synthetic", "judge"


class EjectModelRequest(BaseModel):
    alias: str = "default"


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
    if finetuned_dir.exists():
        for item in finetuned_dir.iterdir():
            if item.is_dir():
                base_model = get_adapter_base_model(str(item))
                models.append(
                    ModelInfo(
                        id=item.name,
                        type="adapter",
                        path=str(item),
                        base_model=base_model,
                    )
                )

    return models


@router.post("/model/load")
async def load_model(request: LoadModelRequest) -> dict[str, str]:
    """
    Load a model into memory.

    When loading an adapter, the base model is auto-detected from adapter_config.json.
    """
    manager = ModelManager.get_instance()
    try:
        adapter_path = None
        base_model_id = request.base_model_id

        if request.adapter_id:
            finetuned_dir = settings.finetuned_models_path
            potential_path = finetuned_dir / request.adapter_id
            if potential_path.exists():
                adapter_path = str(potential_path)
            else:
                adapter_path = request.adapter_id  # Assume absolute or HF ID

            # Auto-detect base model from adapter config
            detected_base = get_adapter_base_model(adapter_path)
            if detected_base:
                if base_model_id != detected_base:
                    logger.info(f"Auto-correcting base model: {base_model_id} -> {detected_base}")
                base_model_id = detected_base

        await manager.load_model(base_model_id, adapter_path, alias=request.alias)
        return {
            "status": "success",
            "message": f"Loaded {base_model_id} (adapter: {adapter_path}) as '{request.alias}'",
        }
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/model/current")
async def get_current_model() -> dict[str, dict[str, str | None]]:
    manager = ModelManager.get_instance()
    from typing import cast

    return cast(dict[str, dict[str, str | None]], manager.loaded_configs)


@router.post("/model/eject")
async def eject_model(request: EjectModelRequest) -> dict[str, str]:
    manager = ModelManager.get_instance()
    try:
        await manager.eject_model(request.alias)
        return {"status": "success", "message": f"Model '{request.alias}' ejected"}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
