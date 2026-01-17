import json
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import get_settings
from src.server.model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()


def _validate_adapter_id(adapter_id: str) -> None:
    """
    FM-36: Validate adapter ID to prevent path traversal attacks.

    Ensures adapter_id is a safe directory name without path components.

    Raises:
        HTTPException: 400 if adapter_id contains path traversal attempts
    """
    # Reject empty IDs
    if not adapter_id or not adapter_id.strip():
        raise HTTPException(status_code=400, detail="Adapter ID cannot be empty")

    # Reject absolute paths
    if Path(adapter_id).is_absolute():
        logger.warning(f"Absolute path rejected as adapter_id: {adapter_id!r}")
        raise HTTPException(status_code=400, detail="Adapter ID cannot be an absolute path")

    # Reject path traversal sequences
    if ".." in adapter_id or adapter_id.startswith("/") or adapter_id.startswith("\\"):
        logger.warning(f"Path traversal rejected in adapter_id: {adapter_id!r}")
        raise HTTPException(status_code=400, detail="Invalid adapter ID format")

    # Only allow safe characters: alphanumeric, hyphen, underscore, dot (for versions)
    # This also allows HuggingFace-style IDs like "username/model-name"
    if not re.match(r"^[a-zA-Z0-9_\-./]+$", adapter_id):
        logger.warning(f"Invalid characters in adapter_id: {adapter_id!r}")
        raise HTTPException(status_code=400, detail="Adapter ID contains invalid characters")


# FM-42: Known safe model presets (from settings.py)
ALLOWED_MODEL_PRESETS = {
    "mistral",
    "mistral-v3",
    "tinyllama",
    "qwen-7b",
    "qwen-4b",
    "qwen-14b",
    "qwen3-8b",
    "qwen3-4b",
}

# FM-42: Regex pattern for valid HuggingFace Hub model IDs
# Format: "organization/model-name" or "organization/model-name-version"
HF_MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-./]+$")


def _validate_base_model_id(base_model_id: str) -> None:
    """
    FM-42: Validate base model ID to prevent loading arbitrary models.

    Allows:
    - Known model presets (mistral, qwen3-4b, etc.)
    - Valid HuggingFace Hub IDs (org/model-name format)

    Raises:
        HTTPException: 400 if base_model_id is invalid
    """
    if not base_model_id or not base_model_id.strip():
        raise HTTPException(status_code=400, detail="Base model ID cannot be empty")

    # Allow known presets
    if base_model_id.lower() in ALLOWED_MODEL_PRESETS:
        return

    # Allow valid HuggingFace Hub IDs
    if HF_MODEL_ID_PATTERN.match(base_model_id):
        # Additional check: reject if it looks like a path traversal
        if ".." in base_model_id:
            logger.warning(f"Path traversal rejected in base_model_id: {base_model_id!r}")
            raise HTTPException(status_code=400, detail="Invalid base model ID format")
        return

    # Reject anything else
    logger.warning(f"Invalid base_model_id rejected: {base_model_id!r}")
    raise HTTPException(
        status_code=400,
        detail=f"Invalid base model ID. Use a preset ({', '.join(sorted(ALLOWED_MODEL_PRESETS))}) "
        "or a valid HuggingFace Hub ID (e.g., 'Qwen/Qwen3-4B').",
    )


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
        _validate_base_model_id(request.base_model_id)  # FM-42: Validate model ID

        adapter_path = None
        base_model_id = request.base_model_id

        if request.adapter_id:
            _validate_adapter_id(request.adapter_id)  # FM-36: Prevent path traversal

            finetuned_dir = settings.finetuned_models_path
            potential_path = finetuned_dir / request.adapter_id

            # FM-36: Only allow loading from finetuned_models_path or HuggingFace Hub IDs
            if potential_path.exists():
                adapter_path = str(potential_path)
            elif "/" in request.adapter_id and not request.adapter_id.startswith("/"):
                # Looks like a HuggingFace Hub ID (e.g., "username/model-name")
                adapter_path = request.adapter_id
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Adapter '{request.adapter_id}' not found in finetuned models directory",
                )

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
