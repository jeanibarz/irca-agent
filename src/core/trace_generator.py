import json

import shortuuid
from guidance import models, gen, select
import torch

from dataset_generation.functions_factory import FunctionsFactory
from core.prompt.function_calling_oneshot import prompt_template as agent_prompt_template
from core.step_factory import create_step_model, StepType

all_available_functions = FunctionsFactory.load_function_variants(version="v1")
MAX_FUNCS = len(all_available_functions)
DEFAULT_WORKSPACE = "function_calling"
DEFAULT_DATASET = "user_query_ds"


class GuidedTraceGenerator:
    def __init__(self, model_name_or_path, llama2_model=None):
        """
        Initialize the TraceGenerator with specified configuration parameters.

        Args:
            config (dict): Configuration parameters including min_funcs, max_funcs, etc.
        """
        self.model_name_or_path = model_name_or_path

        if llama2_model is None:
            llama2_model = models.Transformers(
                model=self.model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
            )
        self.llama2_model = llama2_model

    def _generate_thought(self, lm, trace, prefix="", suffix="", temperature=0.25):
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
        step = create_step_model(
            step_type=StepType.THOUGHT,
            thought=lm["thought"],
            diff=prefix + "Thought: " + lm["thought"] + suffix,
        )
        trace.append(step)
        return lm

    def _generate_thought_missing_function(self, lm, trace, prefix="", suffix=""):
        forced_thought = "I can't find any function that could be helpful to answer user query. I need to abort the Iterative Resolution Cycle and return a final answer."
        lm += prefix + forced_thought + suffix
        step = create_step_model(
            step_type=StepType.THOUGHT,
            thought=forced_thought,
            diff=prefix + "Thought: " + forced_thought + suffix,
        )
        trace.append(step)
        return lm

    def _generate_action_choice(self, lm, trace, prefix="", suffix=""):
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
        step = create_step_model(
            step_type=StepType.ACTION_CHOICE,
            action_choice=lm["action_choice"],
            diff=prefix + "Action choice: " + lm["action_choice"] + suffix,
        )
        trace.append(step)
        return lm

    def _generate_function_call(self, lm, trace, prefix="", suffix="<|wait|>", temperature=0.0):
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
        step = create_step_model(
            step_type=StepType.FUNCTION_CALL,
            fct_name=lm["fct_name"],
            fct_parameters=lm["fct_parameters"],
            diff=prefix
            + 'Call function: {"name": "'
            + lm["fct_name"]
            + '"}, "parameters": '
            + lm["fct_parameters"]
            + suffix,
        )
        trace.append(step)
        return lm

    def _generate_function_output(self, lm, trace, prefix="", suffix="", temperature=1):
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
        step = create_step_model(
            step_type=StepType.FUNCTION_OUTPUT,
            shortuuid=shortuuid_output,
            function_output=lm["function_output"],
            diff=prefix + f"Output[{shortuuid_output}]: " + lm["function_output"] + suffix,
        )
        trace.append(step)
        return lm

    def _generate_final_answer(self, lm, trace, prefix="\n\n### FINAL ANSWER\n", suffix="<|wait|>", temperature=0.5):
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
        step = create_step_model(
            step_type=StepType.FINAL_ANSWER,
            final_answer=lm["final_answer"].rstrip("\n"),
            diff=prefix + lm["final_answer"].rstrip("\n") + suffix,
        )
        trace.append(step)
        return lm

    def _generate_agent_prompt(
        self, lm, trace, available_functions, user_query, agent_scratchpad="", prefix="", suffix=""
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
        step = create_step_model(
            step_type=StepType.INITIAL_PROMPT,
            diff=prefix
            + agent_prompt_template.format(
                available_functions=available_functions,
                user_query=user_query,
                agent_scratchpad=agent_scratchpad,
            )
            + suffix,
        )
        trace.append(step)
        return lm

    def generate_trace_missing_function(self, trace, available_functions, user_query, next_thought_prefix):
        # Instantiate agent prompt
        lm = self._generate_agent_prompt(
            lm=self.llama2_model,
            trace=trace,
            available_functions=available_functions,
            user_query=user_query,
        )

        # Generate synthetic thought
        lm = self._generate_thought_missing_function(lm=lm, trace=trace, prefix=next_thought_prefix)

        # Final answer
        lm = self._generate_final_answer(lm=lm, trace=trace)

        return trace

    def generate_single_trace(self, available_functions, user_query, lm=None, start_step=0, case="nominal"):
        trace = []

        if lm is None:
            # Instantiate agent prompt
            lm = self._generate_agent_prompt(
                lm=self.llama2_model, trace=trace, available_functions=available_functions, user_query=user_query
            )
            if start_step != 0:
                print("Warning: argument `start_step=0` ignored as agent has been reinitialized")
            start_step = 0

        # Iterative resolution cycle, with at most 10 steps
        max_steps = 10
        curr_step = 0  # track number of steps

        # because prompt starts with a new line, we use prefix="" here
        lm = self._generate_thought(lm=lm, trace=trace, prefix="")
        lm = self._generate_action_choice(lm=lm, trace=trace, prefix="\n")
        while (curr_step < max_steps) and ("call" in trace[-1].action_choice.lower()):
            curr_step += 1
            lm = self._generate_function_call(lm=lm, trace=trace, prefix="\n")
            lm = self._generate_function_output(lm=lm, trace=trace, prefix="\n")
            lm = self._generate_thought(lm=lm, trace=trace, prefix="\n")
            lm = self._generate_action_choice(lm=lm, trace=trace, prefix="\n")

        # Final answer
        lm = self._generate_final_answer(lm=lm, trace=trace)

        return trace

    def generate_traces(self, available_functions, user_query, lm=None, start_step=0, case="nominal"):
        traces = []
        trace = []

        if lm is None:
            # Instantiate agent prompt
            lm = self._generate_agent_prompt(
                lm=self.llama2_model,
                trace=trace,
                available_functions=available_functions,
                user_query=user_query,
            )
            if start_step != 0:
                print("Warning: argument `start_step=0` ignored as agent has been reinitialized")
            start_step = 0

        # Iterative resolution cycle, with at most 10 steps
        max_steps = 5
        curr_step = 0  # track number of steps

        next_thought_prefix = ""
        lm = self._generate_thought(
            lm=lm,
            trace=trace,
            prefix="",  # because prompt starts with a new line
        )
        lm = self._generate_action_choice(lm=lm, trace=trace, prefix="\n")
        while (curr_step < max_steps) and ("call" in trace[-1].action_choice.lower()):
            curr_step += 1
            lm = self._generate_function_call(lm=lm, trace=trace, prefix="\n")

            # Branch in another future instance where trace where chosen function is not available
            chosen_function_name = trace[-1]["fct_name"]
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
                trace=trace[
                    0:-3
                ],  # generate a new trace where last Thought, Action Choice, and Function call are removed
                available_functions=alt_available_functions,  # current functions with chosen function removed
                user_query=user_query,
                next_thought_prefix=next_thought_prefix,
            )
            traces.append(alt_trace)
            lm = self._generate_function_output(lm=lm, trace=trace, prefix="\n")
            lm = self._generate_thought(lm=lm, trace=trace, prefix="\n")
            lm = self._generate_action_choice(lm=lm, trace=trace, prefix="\n")

        # Final answer
        lm = self._generate_final_answer(lm=lm, trace=trace)

        traces.append(trace)

        return traces

    def trace_to_str(self, trace):
        # Build agent trace
        trace_str = ""
        for step in trace:
            trace_str += step["diff"]
        return trace_str

    def trace_to_argilla_record(self, available_functions, user_query, trace):
        import argilla as rg

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

    # def validate_info_completeness(self, lm):
    #     """
    #     Performs a self-consistency check to determine if all necessary information is available for function execution.

    #     This function asks a series of questions in different forms to verify the availability of all required information. It then scores the answers based on expected responses. If the score meets or exceeds a specified threshold, the function concludes that the information is sufficient.

    #     :param lm: A non-mutable instance of a language model.
    #     :param threshold: The minimum score required to consider the information as sufficient. Default is 3.
    #     :return: Boolean value indicating whether the information is sufficient (True) or not (False).
    #     """
    #     question_variants = [
    #         {
    #             "question": "Do I have all necessary information to execute the function ?",
    #             "expected_output_no_missing_information": "yes",
    #         },
    #         {
    #             "question": "Is there any required information missing for the function execution?",
    #             "expected_output_no_missing_information": "no",
    #         },
    #         {
    #             "question": "Can I proceed with the function execution with the information currently available?",
    #             "expected_output_no_missing_information": "yes",
    #         },
    #         {
    #             "question": "Have all mandatory parameters for the function been provided?",
    #             "expected_output_no_missing_information": "yes",
    #         },
    #         {
    #             "question": "Is there additional information I need to gather before executing the function?",
    #             "expected_output_no_missing_information": "no",
    #         },
    #     ]
    #     info_completeness_score = 0
    #     for question_variant in question_variants:
    #         lm_with_question = (
    #             lm + "\n" + question_variant["question"] + " (yes/no) => " + select(["yes", "no"], name="choice")
    #         )
    #         if lm_with_question["choice"] == question_variant["expected_output_no_missing_information"]:
    #             info_completeness_score += 1
    #     return info_completeness_score
