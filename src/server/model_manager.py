import logging
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import get_settings

logger = logging.getLogger(__name__)


class ModelManager:
    _instance = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_model: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.current_base_model_id: str | None = None
        self.current_adapter_id: str | None = None

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def load_model(self, base_model_id: str, adapter_path: str | None = None) -> tuple[Any, Any]:
        """
        Load model into memory. optimizing to avoid reloading if already loaded.
        """
        logger.info(f"Request to load model. Base: {base_model_id}, Adapter: {adapter_path}")

        # 1. Check if base model needs reloading
        if self.base_model is None or self.current_base_model_id != base_model_id:
            logger.info(f"Loading base model: {base_model_id}")
            self._load_base_model(base_model_id)
        else:
            logger.info("Base model already loaded.")

        # 2. Load Adapter if requested
        if adapter_path:
            # Full path check or just name check if we want strictness
            if self.current_adapter_id != adapter_path:
                logger.info(f"Loading adapter: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
                self.model.eval()
                self.current_adapter_id = adapter_path
            else:
                logger.info("Adapter already loaded.")
        else:
            # If no adapter requested, use base model as main model
            self.model = self.base_model
            self.current_adapter_id = None

        return self.model, self.tokenizer

    def _load_base_model(self, base_model_id: str) -> None:
        # Free memory if exists
        if self.base_model:
            del self.base_model
            torch.cuda.empty_cache()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
        self.current_base_model_id = base_model_id

    def generate(self, prompt: str, max_new_tokens: int = 4096, temperature: float = 0.7, top_p: float = 1.0) -> str:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip prompt from response
        # Note: robust stripping
        if response.startswith(prompt):
            return str(response[len(prompt) :])
        return str(response)
