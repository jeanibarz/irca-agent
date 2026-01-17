"""
Trace Generator

Main orchestration class for generating agent traces.
Uses the guidance library for constrained generation.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Protocol, cast

import torch
from guidance import models

from src.core.domain import StepType, Trace, create_step
from src.core.prompt.function_calling_oneshot import prompt_template as agent_prompt_template

from .constants import ACTION_CALL_FUNCTION
from .step_generators import (
    generate_action_choice,
    generate_final_answer,
    generate_function_call,
    generate_function_output,
    generate_thought,
    generate_thought_missing_function,
)

if TYPE_CHECKING:
    from guidance import Program

logger = logging.getLogger(__name__)


class LanguageModel(Protocol):
    """Protocol for language model interface."""

    def __add__(self, other: str) -> LanguageModel: ...

    def __getitem__(self, key: str) -> str: ...


class TraceGenerator:
    """
    Generates agent traces using constrained generation.

    This class orchestrates the generation of complete agent traces,
    including thoughts, action choices, function calls, and final answers.

    The generator uses the guidance library to ensure structured output
    that follows the expected format.
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        model: LanguageModel | None = None,
        torch_dtype: torch.dtype | None = None,
        device_map: dict | None = None,
    ):
        """
        Initialize the TraceGenerator.

        Args:
            model_name_or_path: Path to model or HuggingFace model ID
            model: Pre-initialized language model (for testing/reuse)
            torch_dtype: Torch dtype for model loading
            device_map: Device mapping for model placement
        """
        self.model_name_or_path = model_name_or_path

        if model is not None:
            self.model = model
        elif model_name_or_path is not None:
            self.model = self._load_model(
                model_name_or_path,
                torch_dtype=torch_dtype or torch.bfloat16,
                device_map=device_map or {"": 0},
            )
        else:
            raise ValueError("Either model_name_or_path or model must be provided")

    def _load_model(
        self,
        model_name_or_path: str,
        torch_dtype: torch.dtype,
        device_map: dict | str,
    ) -> LanguageModel:
        """Load the language model."""
        logger.info(f"Loading model: {model_name_or_path}")
        return cast(
            LanguageModel,
            models.Transformers(
                model=model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
            ),
        )

    def _generate_initial_prompt(
        self,
        lm: Program,
        trace: Trace,
        available_functions: str,
        user_query: str,
        agent_scratchpad: str = "",
        prefix: str = "",
        suffix: str = "",
    ) -> Program:
        """Generate the initial agent prompt."""
        prompt = agent_prompt_template.format(
            available_functions=available_functions,
            user_query=user_query,
            agent_scratchpad=agent_scratchpad,
        )
        lm += prefix + prompt + suffix

        step = create_step(
            step_type=StepType.INITIAL_PROMPT,
            diff=prefix + prompt + suffix,
        )
        trace.append(step)
        return lm

    def generate_single_trace(
        self,
        available_functions: str,
        user_query: str,
        max_steps: int = 10,
    ) -> Trace:
        """
        Generate a single complete trace.

        Args:
            available_functions: JSON string of available functions
            user_query: The user's query
            max_steps: Maximum number of function call iterations

        Returns:
            Complete trace with all steps
        """
        trace = Trace()

        # Initialize with agent prompt
        lm = self._generate_initial_prompt(
            lm=self.model,
            trace=trace,
            available_functions=available_functions,
            user_query=user_query,
        )

        # Generate initial thought and action choice
        lm = generate_thought(lm=lm, trace=trace, prefix="")
        lm = generate_action_choice(lm=lm, trace=trace, prefix="\n")

        # Iterative resolution cycle
        curr_step = 0
        while curr_step < max_steps and self._should_continue(trace):
            curr_step += 1
            lm = generate_function_call(lm=lm, trace=trace, prefix="\n")
            lm = generate_function_output(lm=lm, trace=trace, prefix="\n")
            lm = generate_thought(lm=lm, trace=trace, prefix="\n")
            lm = generate_action_choice(lm=lm, trace=trace, prefix="\n")

        # Final answer
        lm = generate_final_answer(lm=lm, trace=trace)

        return trace

    def generate_trace_missing_function(
        self,
        available_functions: str,
        user_query: str,
        base_trace: Trace | None = None,
        next_thought_prefix: str = "",
    ) -> Trace:
        """
        Generate a trace where the required function is missing.

        Used for data augmentation - creates training examples where
        the model must acknowledge it cannot complete the task.

        Args:
            available_functions: JSON string of available functions (without the needed one)
            user_query: The user's query
            base_trace: Optional partial trace to continue from
            next_thought_prefix: Prefix for the thought step

        Returns:
            Trace with acknowledgment of missing function
        """
        trace = base_trace or Trace()

        if not base_trace:
            lm = self._generate_initial_prompt(
                lm=self.model,
                trace=trace,
                available_functions=available_functions,
                user_query=user_query,
            )
        else:
            lm = self.model

        # Generate forced thought about missing function
        lm = generate_thought_missing_function(
            lm=lm,
            trace=trace,
            prefix=next_thought_prefix,
        )

        # Final answer
        lm = generate_final_answer(lm=lm, trace=trace)

        return trace

    def generate_traces_with_augmentation(
        self,
        available_functions: str,
        user_query: str,
        max_steps: int = 5,
    ) -> list[Trace]:
        """
        Generate multiple traces with data augmentation.

        For each function call in the trace, creates an alternative trace
        where that function is removed, teaching the model to handle
        missing capabilities.

        Args:
            available_functions: JSON string of available functions
            user_query: The user's query
            max_steps: Maximum steps for the main trace

        Returns:
            List of traces (main trace + augmented alternatives)
        """
        traces = []

        # Generate main trace
        main_trace = self.generate_single_trace(
            available_functions=available_functions,
            user_query=user_query,
            max_steps=max_steps,
        )

        # Generate augmented traces for each function call
        for i, step in enumerate(main_trace.steps):
            if step.type == StepType.FUNCTION_CALL:
                # Create alternative without this function
                alt_trace = self._generate_alternative_trace(
                    main_trace=main_trace,
                    function_call_index=i,
                    available_functions=available_functions,
                    user_query=user_query,
                )
                if alt_trace:
                    traces.append(alt_trace)

        traces.append(main_trace)
        return traces

    def _generate_alternative_trace(
        self,
        main_trace: Trace,
        function_call_index: int,
        available_functions: str,
        user_query: str,
    ) -> Trace | None:
        """Generate an alternative trace without a specific function."""
        try:
            # Get the function that was called
            function_call_step = main_trace.steps[function_call_index]
            chosen_function_name = function_call_step.fct_name

            # Remove that function from available functions
            functions_list = json.loads(available_functions)
            alt_functions = [f for f in functions_list if f["name"] != chosen_function_name]
            alt_functions_json = json.dumps(alt_functions)

            # Create partial trace up to the function call
            partial_trace = Trace()
            # Copy steps before the function call (excluding thought+action+call = last 3)
            for step in main_trace.steps[: function_call_index - 2]:
                partial_trace.append(step)

            # Generate new trace with missing function
            return self.generate_trace_missing_function(
                available_functions=alt_functions_json,
                user_query=user_query,
                base_trace=partial_trace,
                next_thought_prefix="",
            )
        except Exception as e:
            logger.warning(f"Failed to generate alternative trace: {e}")
            return None

    def _should_continue(self, trace: Trace) -> bool:
        """Check if the trace should continue with another iteration."""
        if not trace.steps:
            return False
        last_step = trace.last_step
        if last_step.type != StepType.ACTION_CHOICE:
            return False
        return ACTION_CALL_FUNCTION in last_step.action_choice.lower()

    def trace_to_string(self, trace: Trace) -> str:
        """Convert a trace to a string representation."""
        return str(trace.to_string())


# Backwards compatibility alias
GuidedTraceGenerator = TraceGenerator
