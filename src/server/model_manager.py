import asyncio
import logging
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import get_settings

logger = logging.getLogger(__name__)


class ModelManager:
    _instance = None

    def __init__(self) -> None:
        self.settings = get_settings()
        # Storage for multiple models {alias: model_obj}
        self.models: dict[str, Any] = {}
        self.tokenizers: dict[str, Any] = {}
        self.base_models: dict[str, Any] = {}

        # Track what is loaded where {alias: {"base": id, "adapter": path}}
        self.loaded_configs: dict[str, dict[str, str | None]] = {}

        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    async def load_model(
        self, base_model_id: str, adapter_path: str | None = None, alias: str = "default"
    ) -> tuple[Any, Any]:
        """
        Load model into memory under a specific alias.
        """
        from src.server.events import EventBroadcaster

        broadcaster = EventBroadcaster.get_instance()

        if self._lock.locked():
            raise RuntimeError("Model loading already in progress.")

        async with self._lock:
            # Emit Start
            broadcaster.publish(
                "model_progress", {"alias": alias, "step": "init", "message": "Initializing request..."}
            )
            logger.info(f"Request to load model [{alias}]. Base: {base_model_id}, Adapter: {adapter_path}")

            # Check if this specific configuration is already loaded for this alias
            current_config = self.loaded_configs.get(alias)
            if current_config and current_config["base"] == base_model_id and current_config["adapter"] == adapter_path:
                logger.info(f"Model [{alias}] already loaded with this config.")
                broadcaster.publish(
                    "model_progress", {"alias": alias, "step": "done", "message": "Model already loaded."}
                )
                return self.models[alias], self.tokenizers[alias]

            # 1. Load Base Model (or reuse if available)
            base_model_obj = self._find_existing_base_model(base_model_id)
            tokenizer_obj = self._find_existing_tokenizer(base_model_id)

            if not base_model_obj:
                broadcaster.publish(
                    "model_progress",
                    {"alias": alias, "step": "download_base", "message": "Downloading/Loading base model..."},
                )
                logger.info(f"Loading new base model: {base_model_id}")

                # Run in thread with stderr interception (inside the method)
                # We do NOT use _run_with_heartbeat here because it conflicts with the granular tqdm updates
                base_model_obj, tokenizer_obj = await asyncio.to_thread(
                    self._load_base_model_and_tokenizer,
                    base_model_id,
                    broadcaster,
                    alias,
                )
                self.base_models[base_model_id] = base_model_obj
            else:
                broadcaster.publish(
                    "model_progress",
                    {"alias": alias, "step": "reuse_base", "message": "Reusing cached base model..."},
                )
                logger.info(f"Reusing existing base model instance for: {base_model_id}")
                if not tokenizer_obj:
                    tokenizer_obj = AutoTokenizer.from_pretrained(base_model_id)
                    tokenizer_obj.pad_token = tokenizer_obj.eos_token
                    tokenizer_obj.padding_side = "right"

            # 2. Load Adapter if requested
            model_to_store = base_model_obj
            if adapter_path:
                broadcaster.publish(
                    "model_progress",
                    {
                        "alias": alias,
                        "step": "load_adapter",
                        "message": f"Loading adapter: {Path(adapter_path).name}...",
                    },
                )
                logger.info(f"Loading adapter: {adapter_path}")
                model_to_store = await self._run_with_heartbeat(
                    broadcaster,
                    alias,
                    "Loading adapter weights",
                    PeftModel.from_pretrained,
                    base_model_obj,
                    adapter_path,
                )
                model_to_store.eval()

            # Store references
            self.models[alias] = model_to_store
            self.tokenizers[alias] = tokenizer_obj

            # Update config tracking
            self.loaded_configs[alias] = {"base": base_model_id, "adapter": adapter_path}

            broadcaster.publish(
                "model_progress", {"alias": alias, "step": "done", "message": "Model loaded successfully."}
            )
            return self.models[alias], self.tokenizers[alias]

    def _find_existing_base_model(self, base_model_id: str) -> Any | None:
        return self.base_models.get(base_model_id)

    def _find_existing_tokenizer(self, base_model_id: str) -> Any | None:
        # Heuristic: Check other aliases
        for alias, config in self.loaded_configs.items():
            if config["base"] == base_model_id:
                return self.tokenizers.get(alias)
        return None

    async def _run_with_heartbeat(
        self,
        broadcaster: Any,
        alias: str,
        activity_name: str,
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Run a blocking function in a thread while emitting periodic heartbeat messages.
        This provides continuous feedback to the user during long-running operations.
        """
        import time

        start_time = time.time()
        heartbeat_task = None
        result_container: list[Any] = []
        error_container: list[Exception] = []
        done_event = asyncio.Event()

        async def heartbeat() -> None:
            """Emit periodic status updates."""
            elapsed = 0
            while not done_event.is_set():
                await asyncio.sleep(2)  # Update every 2 seconds
                if done_event.is_set():
                    break
                elapsed = int(time.time() - start_time)
                broadcaster.publish(
                    "model_progress",
                    {
                        "alias": alias,
                        "step": "loading",
                        "message": f"{activity_name}... ({elapsed}s elapsed)",
                    },
                )

        def blocking_call() -> None:
            try:
                result_container.append(func(*args, **kwargs))
            except Exception as e:
                error_container.append(e)

        # Start heartbeat
        heartbeat_task = asyncio.create_task(heartbeat())

        # Run blocking call in thread
        await asyncio.to_thread(blocking_call)

        # Stop heartbeat
        done_event.set()
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        if error_container:
            raise error_container[0]

        return result_container[0]

    async def eject_model(self, alias: str = "default") -> None:
        from src.server.events import EventBroadcaster

        broadcaster = EventBroadcaster.get_instance()

        async with self._lock:
            if alias in self.models:
                del self.models[alias]
            if alias in self.tokenizers:
                del self.tokenizers[alias]
            if alias in self.loaded_configs:
                del self.loaded_configs[alias]

            torch.cuda.empty_cache()
            broadcaster.publish("model_progress", {"alias": alias, "step": "ejected", "message": "Model ejected."})
            logger.info(f"Model [{alias}] ejected.")

    def _load_base_model_and_tokenizer(
        self, base_model_id: str, broadcaster: Any = None, alias: str = "default"
    ) -> tuple[Any, Any]:
        """Load fresh base model and tokenizer with progress feedback via stderr interception."""
        import io
        import re
        import sys
        import threading
        import time

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Force transformers/HF hub to show progress bars even if non-interactive
        from transformers.utils import logging as hf_logging

        hf_logging.enable_progress_bar()

        # Load tokenizer (quick)
        if broadcaster:
            broadcaster.publish(
                "model_progress",
                {"alias": alias, "step": "tokenizer", "message": "Loading tokenizer..."},
            )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load model with stderr interception to capture tqdm progress
        if broadcaster:
            broadcaster.publish(
                "model_progress",
                {"alias": alias, "step": "model_shards", "message": "Loading model shards..."},
            )

        # Create a custom stderr wrapper that captures tqdm output AND supports timeout
        class TqdmCapture(io.StringIO):
            def __init__(self, original_stderr: Any, bc: Any, al: str) -> None:
                super().__init__()
                self.original = original_stderr
                self.broadcaster = bc
                self.alias = al
                self.last_pct = -1
                self.pattern = re.compile(r"(\d+)%\|")
                self.last_update = 0.0

            def isatty(self) -> bool:
                return True  # Pretend to be a TTY to encourage progress bars

            def write(self, text: str) -> int:
                self.original.write(text)
                self.last_update = time.time()  # Mark activity

                if self.broadcaster and "%" in text:
                    match = self.pattern.search(text)
                    if match:
                        pct = int(match.group(1))
                        if pct != self.last_pct:
                            self.last_pct = pct
                            self.broadcaster.publish(
                                "model_progress",
                                {
                                    "alias": self.alias,
                                    "step": "loading_shards",
                                    "message": f"Loading: {pct}%",
                                    "progress": pct,
                                },
                            )
                return len(text)

            def flush(self) -> None:
                self.original.flush()

        # Capture stderr during model loading
        original_stderr = sys.stderr
        capture = TqdmCapture(original_stderr, broadcaster, alias)

        # Start a background thread to emit "Downloading/Processing" if silenced
        stop_fallback = threading.Event()

        def fallback_monitor() -> None:
            import time

            start = time.time()
            while not stop_fallback.is_set():
                time.sleep(1)
                now = time.time()
                # If no activity for 2 seconds, and total time > 3s (to avoid noisy start)
                if (now - capture.last_update) > 2.0 and (now - start) > 2.0:
                    if broadcaster:
                        elapsed = int(now - start)
                        broadcaster.publish(
                            "model_progress",
                            {
                                "alias": alias,
                                "step": "downloading",
                                "message": f"Downloading/Processing... ({elapsed}s elapsed)",
                            },
                        )

        monitor_thread = threading.Thread(target=fallback_monitor, daemon=True)
        monitor_thread.start()

        try:
            sys.stderr = capture
            # Need to initialize time inside capture for reference
            import time

            capture.last_update = time.time()

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        finally:
            stop_fallback.set()
            sys.stderr = original_stderr
            monitor_thread.join(timeout=1.0)

        return base_model, tokenizer

    async def generate(
        self,
        prompt: str,
        alias: str = "default",
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop_tokens: list[str] | None = None,
    ) -> str:
        model = self.models.get(alias)
        tokenizer = self.tokenizers.get(alias)

        if not model or not tokenizer:
            # Fallback for usability: if only 1 model loaded, use it?
            if len(self.models) == 1:
                fallback_alias = list(self.models.keys())[0]
                logger.info(f"Alias '{alias}' not found, falling back to '{fallback_alias}'")
                model = self.models[fallback_alias]
                tokenizer = self.tokenizers[fallback_alias]
            else:
                # If synthetic requested but not found, and we have default, use default
                if alias == "synthetic" and "default" in self.models:
                    model = self.models["default"]
                    tokenizer = self.tokenizers["default"]
                else:
                    raise RuntimeError(f"Model alias '{alias}' not available. Loaded: {list(self.models.keys())}")

        # Run generation in thread to avoid blocking event loop
        return await asyncio.to_thread(
            self._generate_sync, model, tokenizer, prompt, max_new_tokens, temperature, top_p, stop_tokens
        )  # type: ignore[no-any-return]

    # Default stop sequences for IRCA agent format
    DEFAULT_STOP_SEQUENCES = [
        "### USER QUERY",  # Stop before generating fake user queries
        "### INSTRUCTIONS",  # Stop before regenerating instructions
        "<|wait|>",  # Qwen wait token
        "<|im_end|>",  # ChatML end token
    ]

    def _generate_sync(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_tokens: list[str] | None = None,
    ) -> str:
        from transformers import StoppingCriteriaList

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Merge default and custom stop sequences
        all_stop_sequences = list(self.DEFAULT_STOP_SEQUENCES)
        if stop_tokens:
            all_stop_sequences.extend(stop_tokens)

        # Create stopping criteria for early termination
        # Pass prompt length so we only check GENERATED tokens, not the prompt
        prompt_length = inputs.input_ids.shape[1]
        stopping_criteria = StoppingCriteriaList([StopOnSequences(tokenizer, all_stop_sequences, prompt_length)])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )

        # Slice to keep only new tokens
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up: truncate at stop sequence (in case partial match)
        response = self._truncate_at_stop_sequences(response, all_stop_sequences)

        return str(response)

    def _truncate_at_stop_sequences(self, text: str, stop_sequences: list[str]) -> str:
        """Truncate text at the first occurrence of any stop sequence."""
        earliest_pos = len(text)
        for stop_seq in stop_sequences:
            pos = text.find(stop_seq)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
        return text[:earliest_pos].strip()


class StopOnSequences:
    """Stopping criteria that stops generation when any stop sequence is detected."""

    def __init__(self, tokenizer: Any, stop_sequences: list[str], prompt_length: int):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.prompt_length = prompt_length  # Track where generation starts
        self.call_count = 0
        # Pre-compute stop sequence token patterns for efficiency
        self.stop_patterns: list[tuple[str, list[int]]] = []
        for seq in stop_sequences:
            tokens = tokenizer.encode(seq, add_special_tokens=False)
            if tokens:
                self.stop_patterns.append((seq, tokens))
                logger.info(f"Stop pattern: {seq!r} -> {tokens}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        self.call_count += 1
        # Only check GENERATED tokens (after prompt)
        generated_length = input_ids.shape[1] - self.prompt_length

        if self.call_count == 1:
            logger.info(
                f"StopOnSequences first call: prompt_length={self.prompt_length}, total_length={input_ids.shape[1]}"
            )

        if generated_length < 5:
            return False  # Don't stop too early

        # Check the last N tokens against stop patterns
        for seq_text, pattern in self.stop_patterns:
            pattern_len = len(pattern)
            if generated_length >= pattern_len:
                last_tokens = input_ids[0, -pattern_len:].tolist()
                if last_tokens == pattern:
                    logger.info(f"STOP: Token pattern match for {seq_text!r}")
                    return True

        # Check decoded text every 20 tokens
        if generated_length > 20 and generated_length % 20 == 0:
            # Decode only the GENERATED tokens
            generated_tokens = input_ids[0, self.prompt_length :]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if self.call_count % 100 == 0:
                logger.info(f"Generated {generated_length} tokens, text ends with: ...{generated_text[-100:]!r}")

            for seq in self.stop_sequences:
                if seq in generated_text:
                    logger.info(f"STOP: Text match for {seq!r}")
                    return True

        return False
