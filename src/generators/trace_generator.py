import json
import random
from collections import defaultdict
from random import shuffle

import numpy as np
import json
import shortuuid

from openai import OpenAI
from guidance import models, gen, select
import torch

from function_calling.ds_generation.gpt4_functions_v2 import all_available_functions

# from function_calling.ds_generation.glaive_v2_functions import all_available_functions
from prompt.generate_user_query import prompt_template as user_query_prompt_template
from prompt.function_calling_oneshot import prompt_template as agent_prompt_template
from utils import format_generate_user_query

# from prompt.function_calling_oneshot_chatml import (
#     prompt_template as agent_prompt_template,
# )

MAX_FUNCS = len(all_available_functions)
DEFAULT_WORKSPACE = "function_calling"
DEFAULT_DATASET = "user_query_ds"


class GuidedTraceGenerator:
    def __init__(self, config, llama2_model=None, trace=None):
        """
        Initialize the TraceGenerator with specified configuration parameters.

        Args:
            config (dict): Configuration parameters including min_funcs, max_funcs, etc.
        """
        # Set default values and update with any provided configuration
        default_config = {
            "model_name_or_path": "/workspace/models/TheBloke/Mistral-7B-Instruct-v0.2-DARE-GGUF/mistral-7b-instruct-v0.2-dare.Q3_K_L.gguf",
            "min_funcs": 1,
            "max_funcs": MAX_FUNCS,
            "size": 1,  # The number of functions to be sampled, typically 1
            "distribution": {
                "name": "beta",  # Using the beta distribution
                "parameters": {
                    "alpha": 1.4,
                    "beta": 5,
                },
            },
        }
        config = {**default_config, **config}  # Update defaults with provided config

        # Ensure the maximum number of functions does not exceed available functions
        if config["max_funcs"] > len(all_available_functions):
            raise ValueError(
                f"max_funcs cannot exceed the total number of available functions ({MAX_FUNCS})."
            )

        if llama2_model is None:
            llama2_model = models.Transformers(
                model=config["model_name_or_path"],
                # n_ctx=4096,
                # n_gpu_layers=32,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
            )

        if trace is None:
            trace = []

        self.config = config
        self.use_gpt4 = False
        self.llama2_model = llama2_model
        self.min_funcs = config["min_funcs"]
        self.max_funcs = config["max_funcs"]
        self.size = config["size"]
        self.distribution_config = config["distribution"]
        self.rg = config.get("rg", None)
        self.rg_ds_name = config.get("argilla_target", {}).get("name", None)
        self.rg_workspace = config.get("argilla_target", {}).get("workspace", None)
        self.trace = trace

    def _shuffle_json_functions(available_functions):
        # Shuffle the list of functions
        functions = json.loads(available_functions)
        random.shuffle(functions)

        available_functions = json.dumps(functions)
        return available_functions

    def _generate_thought(
        self, lm, prefix="", suffix="", temperature=0.25, trace_to_append=None
    ):
        lm += (
            prefix
            + "Thought: "
            + gen(
                max_tokens=500,
                name="thought",
                stop="\n",
                temperature=temperature,
            )
            + suffix
        )
        step = dict(
            type="thought",
            thought=lm["thought"],
            diff=prefix + "Thought: " + lm["thought"] + suffix,
        )
        if trace_to_append is None:
            self.trace.append(step)
        else:
            trace_to_append.append(step)
        return lm

    def _generate_thought_missing_function(
        self, lm, prefix="", suffix="", trace_to_append=None
    ):
        forced_thought = "I can't find any function that could be helpful to answer user query. I need to abort the Iterative Resolution Cycle and return a final answer."
        lm += prefix + forced_thought + suffix
        step = dict(
            type="thought",
            thought=forced_thought,
            diff=prefix + "Thought: " + forced_thought + suffix,
        )
        if trace_to_append is None:
            self.trace.append(step)
        else:
            trace_to_append.append(step)
        return lm

    def _generate_action_choice(self, lm, prefix="", suffix="", trace_to_append=None):
        lm += (
            prefix
            + "Action choice: "
            + select(
                options=[
                    "call function",
                    "final answer",
                ],
                name="action_choice",
            )
            + suffix
        )
        step = dict(
            type="action_choice",
            action_choice=lm["action_choice"],
            diff=prefix + "Action choice: " + lm["action_choice"] + suffix,
        )
        if trace_to_append is None:
            self.trace.append(step)
        else:
            trace_to_append.append(step)
        return lm

    def _generate_function_call(
        self, lm, prefix="", suffix="<|wait|>", temperature=0.0, trace_to_append=None
    ):
        lm += (
            prefix
            + 'Call function: {"name": "'
            + gen("fct_name", stop='"')
            + '"}, "parameters": '
            + gen(
                max_tokens=500,
                name="fct_parameters",
                stop=["\n", "<|wait|>"],
                temperature=temperature,
            )
            + suffix
        )
        step = dict(
            type="function_call",
            fct_name=lm["fct_name"],
            fct_parameters=lm["fct_parameters"],
            diff=prefix
            + 'Call function: {"name": "'
            + lm["fct_name"]
            + '"}, "parameters": '
            + lm["fct_parameters"]
            + suffix,
        )
        if trace_to_append is None:
            self.trace.append(step)
        else:
            trace_to_append.append(step)
        return lm

    def _generate_function_output(
        self, lm, prefix="", suffix="", temperature=1, trace_to_append=None
    ):
        shortuuid_output = shortuuid.uuid()
        lm += (
            prefix
            + f"Output[{shortuuid_output}]: "
            + gen(
                max_tokens=500,
                name="function_output",
                stop="\n",
                temperature=temperature,
            )
            + suffix
        )
        step = dict(
            type="function_output",
            shortuuid=shortuuid,
            function_output=lm["function_output"],
            diff=prefix
            + f"Output[{shortuuid_output}]: "
            + lm["function_output"]
            + suffix,
        )
        if trace_to_append is None:
            self.trace.append(step)
        else:
            trace_to_append.append(step)
        return lm

    def _generate_final_answer(
        self,
        lm,
        prefix="\n\n### FINAL ANSWER\n",
        suffix="<|wait|>",
        temperature=0.5,
        trace_to_append=None,
    ):
        lm += (
            prefix
            + gen(
                max_tokens=500,
                name="final_answer",
                temperature=temperature,
                stop=["### INSTRUCTIONS", "### USER QUERY", "<|wait|>"],
            )
            + suffix
        )
        step = dict(
            type="final_answer",
            final_answer=lm["final_answer"].rstrip("\n"),
            diff=prefix + lm["final_answer"].rstrip("\n") + suffix,
        )
        if trace_to_append is None:
            self.trace.append(step)
        else:
            trace_to_append.append(step)
        return lm

    def _generate_agent_prompt(
        self,
        lm,
        available_functions,
        user_query,
        agent_scratchpad="",
        prefix="",
        suffix="",
        trace_to_append=None,
    ):
        lm += (
            prefix
            + agent_prompt_template.format(
                available_functions=available_functions,
                user_query=user_query,
                agent_scratchpad=agent_scratchpad,
            )
            + suffix
        )
        step = dict(
            type="initial_prompt",
            diff=prefix
            + agent_prompt_template.format(
                available_functions=available_functions,
                user_query=user_query,
                agent_scratchpad=agent_scratchpad,
            )
            + suffix,
        )
        if trace_to_append is None:
            self.trace.append(step)
        else:
            trace_to_append.append(step)
        return lm

    def generate_trace_missing_function(
        self, available_functions, user_query, next_thought_prefix
    ):
        # Expect the trace to be empty
        assert not len(self.trace)

        # Instantiate agent prompt
        lm = self._generate_agent_prompt(
            lm=self.llama2_model,
            available_functions=available_functions,
            user_query=user_query,
        )

        # Generate synthetic thought
        lm = self._generate_thought_missing_function(
            lm,
            prefix=next_thought_prefix,
        )

        # Final answer
        lm = self._generate_final_answer(lm=lm)

        return self.trace

    def generate_single_trace(
        self, available_functions, user_query, lm=None, start_step=0, case="nominal"
    ):
        trace = []

        if lm is None:
            # Instantiate agent prompt
            lm = self._generate_agent_prompt(
                lm=self.llama2_model,
                available_functions=available_functions,
                user_query=user_query,
                trace_to_append=trace,
            )
            if start_step != 0:
                print(
                    "Warning: argument `start_step=0` ignored as agent has been reinitialized"
                )
            start_step = 0

        # Iterative resolution cycle, with at most 10 steps
        max_steps = 5
        curr_step = 0  # track number of steps

        # because prompt starts with a new line, we use prefix="" here
        lm = self._generate_thought(lm=lm, prefix="", trace_to_append=trace)
        lm = self._generate_action_choice(lm=lm, prefix="\n", trace_to_append=trace)
        while (curr_step < max_steps) and (
            "call" in trace[-1]["action_choice"].lower()
        ):
            curr_step += 1
            lm = self._generate_function_call(lm=lm, prefix="\n", trace_to_append=trace)
            lm = self._generate_function_output(
                lm=lm, prefix="\n", trace_to_append=trace
            )
            lm = self._generate_thought(lm=lm, prefix="\n", trace_to_append=trace)
            lm = self._generate_action_choice(lm=lm, prefix="\n", trace_to_append=trace)

        # Final answer
        lm = self._generate_final_answer(lm=lm, trace_to_append=trace)

        return trace

    def generate_traces(
        self, available_functions, user_query, lm=None, start_step=0, case="nominal"
    ):
        traces = []

        if lm is None:
            # Instantiate agent prompt
            lm = self._generate_agent_prompt(
                lm=self.llama2_model,
                available_functions=available_functions,
                user_query=user_query,
            )
            if start_step != 0:
                print(
                    "Warning: argument `start_step=0` ignored as agent has been reinitialized"
                )
            start_step = 0

        # Iterative resolution cycle, with at most 10 steps
        max_steps = 5
        curr_step = 0  # track number of steps

        next_thought_prefix = ""
        lm = self._generate_thought(
            lm=lm,
            prefix="",  # because prompt starts with a new line
        )
        lm = self._generate_action_choice(lm=lm, prefix="\n")
        while (curr_step < max_steps) and (
            "call" in self.trace[-1]["action_choice"].lower()
        ):
            curr_step += 1
            lm = self._generate_function_call(lm=lm, prefix="\n")

            # Branch in another future instance where trace where chosen function is not available
            chosen_function_name = self.trace[-1]["fct_name"]
            alt_available_functions = json.dumps(
                [
                    function_dict
                    for function_dict in json.loads(available_functions)
                    if function_dict["name"] != chosen_function_name
                ]
            )
            alt_generator = GuidedTraceGenerator(
                self.config,
                llama2_model=self.llama2_model,  # use same LLM instance
            )
            alt_trace = alt_generator.generate_trace_missing_function(
                available_functions=alt_available_functions,  # current functions with chosen function removed
                user_query=user_query,
                next_thought_prefix=next_thought_prefix,
            )
            traces.append(alt_trace)
            lm = self._generate_function_output(lm=lm, prefix="\n")
            next_thought_prefix = "\n"
            lm = self._generate_thought(lm=lm, prefix="\n")
            lm = self._generate_action_choice(lm=lm, prefix="\n")

        # Final answer
        lm = self._generate_final_answer(lm=lm)

        traces.append(self.trace)

        return traces

    def trace_to_str(self, trace):
        # Build agent trace
        trace_str = ""
        for step in trace:
            trace_str += step["diff"]
        return trace_str

    def trace_to_argilla_record(self, available_functions, user_query, trace=None):
        import argilla as rg

        if trace is None:
            trace = self.trace

        # Create record
        print("Creating argilla record for generated user query...")
        record = rg.FeedbackRecord(
            fields={
                "available_functions": available_functions,
                "user_query": user_query,
                "agent_trace": self.trace_to_str(trace),
            },
            # responses=[
            #     {
            #         "values": {
            #             "corrected_user_query": {
            #                 "value": user_query,
            #             },
            #             "corrected_irca_agent_trace": {
            #                 "value": trace,
            #             },
            #         },
            #     },
            # ],
        )
        return record

    def validate_info_completeness(self, lm):
        """
        Performs a self-consistency check to determine if all necessary information is available for function execution.

        This function asks a series of questions in different forms to verify the availability of all required information. It then scores the answers based on expected responses. If the score meets or exceeds a specified threshold, the function concludes that the information is sufficient.

        :param lm: A non-mutable instance of a language model.
        :param threshold: The minimum score required to consider the information as sufficient. Default is 3.
        :return: Boolean value indicating whether the information is sufficient (True) or not (False).
        """
        question_variants = [
            {
                "question": "Do I have all necessary information to execute the function ?",
                "expected_output_no_missing_information": "yes",
            },
            {
                "question": "Is there any required information missing for the function execution?",
                "expected_output_no_missing_information": "no",
            },
            {
                "question": "Can I proceed with the function execution with the information currently available?",
                "expected_output_no_missing_information": "yes",
            },
            {
                "question": "Have all mandatory parameters for the function been provided?",
                "expected_output_no_missing_information": "yes",
            },
            {
                "question": "Is there additional information I need to gather before executing the function?",
                "expected_output_no_missing_information": "no",
            },
        ]
        info_completeness_score = 0
        for question_variant in question_variants:
            lm_with_question = (
                lm
                + "\n"
                + question_variant["question"]
                + " (yes/no) => "
                + select(["yes", "no"], name="choice")
            )
            if (
                lm_with_question["choice"]
                == question_variant["expected_output_no_missing_information"]
            ):
                info_completeness_score += 1
        return info_completeness_score


class FreeTraceGenerator:
    def __init__(self, config, llama2_model=None, trace=None):
        """
        Initialize the TraceGenerator with specified configuration parameters.

        Args:
            config (dict): Configuration parameters including min_funcs, max_funcs, etc.
        """
        # Set default values and update with any provided configuration
        default_config = {
            "model_name_or_path": "/workspace/models/TheBloke/Mistral-7B-Instruct-v0.2-DARE-GGUF/mistral-7b-instruct-v0.2-dare.Q3_K_L.gguf",
            "min_funcs": 1,
            "max_funcs": MAX_FUNCS,
            "size": 1,  # The number of functions to be sampled, typically 1
            "distribution": {
                "name": "beta",  # Using the beta distribution
                "parameters": {
                    "alpha": 1.4,
                    "beta": 5,
                },
            },
        }
        config = {**default_config, **config}  # Update defaults with provided config

        # Ensure the maximum number of functions does not exceed available functions
        if config["max_funcs"] > len(all_available_functions):
            raise ValueError(
                f"max_funcs cannot exceed the total number of available functions ({MAX_FUNCS})."
            )

        if llama2_model is None:
            llama2_model = models.Transformers(
                model=config["model_name_or_path"],
                # n_ctx=4096,
                # n_gpu_layers=32,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
            )

        if trace is None:
            trace = []

        self.config = config
        self.use_gpt4 = False
        self.llama2_model = llama2_model
        self.min_funcs = config["min_funcs"]
        self.max_funcs = config["max_funcs"]
        self.size = config["size"]
        self.distribution_config = config["distribution"]
        self.rg = config.get("rg", None)
        self.rg_ds_name = config.get("argilla_target", {}).get("name", None)
        self.rg_workspace = config.get("argilla_target", {}).get("workspace", None)
        self.trace = trace

    def _shuffle_json_functions(available_functions):
        # Shuffle the list of functions
        functions = json.loads(available_functions)
        random.shuffle(functions)

        available_functions = json.dumps(functions)
        return available_functions
