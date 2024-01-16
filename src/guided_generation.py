import json
import random
from collections import defaultdict

import numpy as np
import json
import shortuuid

from guidance import models, gen, select
import torch
from src.function_calling.ds_generation.gpt4_functions_v1 import all_available_functions
from prompt.generate_user_query import prompt_template as user_query_prompt_template

from prompt.function_calling_oneshot import prompt_template as agent_prompt_template

# from prompt.function_calling_oneshot_chatml import (
#     prompt_template as agent_prompt_template,
# )

MAX_FUNCS = len(all_available_functions)
DEFAULT_WORKSPACE = "function_calling"
DEFAULT_DATASET = "user_query_ds"


class GuidedPromptGenerator:
    def __init__(self, config):
        """
        Initialize the PromptGenerator with specified configuration parameters.

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

        model_name_or_path = config["model_name_or_path"]
        # self.llama2_model = models.LlamaCpp(
        #     model=model_name_or_path,
        #     n_ctx=4096,
        #     n_gpu_layers=32,
        #     device_map={"": 0},
        # )
        self.llama2_model = models.Transformers(
            model=model_name_or_path,
            # n_ctx=4096,
            # n_gpu_layers=32,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )

        self.min_funcs = config["min_funcs"]
        self.max_funcs = config["max_funcs"]
        self.size = config["size"]
        self.distribution_config = config["distribution"]
        self.rg = config.get("rg", None)
        self.rg_ds_name = config.get("argilla", {}).get("name", None)
        self.rg_workspace = config.get("argilla", {}).get("workspace", None)

        # Define lambda functions or regular functions for distributions
        self.distributions = {
            "uniform": lambda size: np.random.uniform(
                low=0,
                high=1,
                size=size,
            ),
            # For assistance in selecting alpha and beta parameters for the Beta distribution,
            # consider using the interactive Beta distribution plot tool available at:
            # https://homepage.stat.uiowa.edu/~mbognar/applets/beta.html
            "beta": lambda alpha, beta, size: np.random.beta(
                a=alpha,
                b=beta,
                size=size,
            ),
        }

    def _scale_to_interval(self, samples):
        """
        Private method to scale a set of numbers from [0, 1] to the specified interval [min_funcs, max_funcs].

        Args:
            samples (numpy.ndarray): The raw samples to scale, assumed to be in [0, 1].

        Returns:
            numpy.ndarray: The scaled samples.
        """
        # Scale samples from [0, 1] to [0, max_funcs - min_funcs]
        scaled_samples = samples * (self.max_funcs - self.min_funcs)
        # Translate samples to [min_funcs, max_funcs]
        interval_samples = scaled_samples + self.min_funcs
        return np.round(interval_samples).astype(int)

    def generate_random_subset(self, return_json=False):
        """
        Generate a random subset of functions using the specified distribution function.

        Args:
            return_json (bool): If True, return functions in JSON format. Defaults to False.

        Returns:
            list or str: A subset of the available functions or JSON string of the same.
        """
        distribution_name = self.distribution_config["name"]
        distribution_params = self.distribution_config["parameters"]
        distribution_func = self.distributions[distribution_name]
        raw_samples = distribution_func(size=self.size, **distribution_params)
        scaled_samples = self._scale_to_interval(raw_samples)
        num_functions = min(int(scaled_samples[0]), len(all_available_functions))

        selected_functions = random.sample(all_available_functions, num_functions)

        if return_json:
            return json.dumps(selected_functions)  # Convert list to JSON string
        else:
            return selected_functions  # Return list directly

    def generate_irca_agent_trace(self, available_functions, satisfiable=True):
        # client = OpenAI()

        user_query_prompt = user_query_prompt_template.format(
            available_functions=available_functions,
            does_or_does_not="does" if satisfiable else "does not",
        )

        # print("prompt after instantiation:")
        # print(user_query_prompt)

        workflow = []

        # Generate user query
        lm = (
            self.llama2_model
            + user_query_prompt
            + gen(
                max_tokens=500,
                name="query",
                stop="\n",
                temperature=1,
            )
        )
        user_query = lm["query"]
        workflow.append(("user_query", lm["query"]))
        # Generate agent thought and function calls iteratively
        lm = (
            self.llama2_model
            + agent_prompt_template.format(
                available_functions=available_functions,
                user_query=user_query,
                agent_scratchpad="Thought: ",
            )
            + gen(
                max_tokens=500,
                name="thought",
                stop="\n",
                temperature=0.25,
            )
        )
        workflow.append(("thought", lm["thought"]))
        i = 0
        while i < 10:
            i += 1
            temp_lm = (
                lm
                + " Next, I will "
                + select(
                    options=[
                        "call a function.",
                        "ask the user for more information.",
                        "give a final answer.",
                        "abort because I don't have the capabilities to fulfill the user request.",
                    ],
                    name="action_choice",
                )
            )
            workflow.append(("action_choice", temp_lm["action_choice"]))
            if temp_lm["action_choice"] == "call a function.":
                lm += (
                    "\nCall function: "
                    + gen(
                        max_tokens=500,
                        name="call_function",
                        stop="\n",
                        temperature=0,
                    )
                    + "\n<wait_for_output>"
                )
                workflow.append(("call_function", lm["call_function"]))

                # Generate synthetic function output
                output_shortuuid = shortuuid.uuid()
                lm += "\nOutput[{output_uuid}]: " + gen(
                    max_tokens=500,
                    name="function_output",
                    stop="\n",
                    temperature=1,
                )
                workflow.append(
                    (
                        "function_output",
                        output_shortuuid,
                        lm["function_output"],
                    )
                )

                # Genere new thought
                lm += "\nThought: " + gen(
                    max_tokens=500,
                    name="thought",
                    stop="\n",
                    temperature=0.25,
                )
                workflow.append(("thought", lm["thought"]))
            else:
                break
        lm += "\n\n### FINAL ANSWER\n" + gen(
            max_tokens=500,
            name="final_answer",
            stop="\n",
            temperature=0.25,
        )
        workflow.append(("final_answer", lm["final_answer"]))

        # print(json.dumps(workflow, indent=4))

        # Build agent trace
        trace = agent_prompt_template.format(
            available_functions=available_functions,
            user_query=workflow[0][1],
            agent_scratchpad="Thought: " + workflow[1][1],
        )

        for output in workflow[2:]:
            if output[0] == "thought":
                trace += "\nThought: " + output[1]
            elif output[0] == "call_function":
                trace += "\nCall function: " + output[1] + "\n<wait_for_output>"
            elif output[0] == "function_output":
                trace += f"\nOutput[{output[1]}]: " + output[2]
            elif output[0] == "final_answer":
                trace += f"\n\n### FINAL ANSWER\n" + output[1]
        # print("-" * 20)
        # print(trace)

        # Create record
        if self.rg:
            print("Creating argilla record for generated user query...")
            record = self.rg.FeedbackRecord(
                fields={
                    # "available_functions": available_functions,
                    # "original_user_query": user_query,
                    "original_irca_agent_trace": trace,
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

            # Push record to argilla dataset
            ds = self.rg.FeedbackDataset.from_argilla(
                name=self.rg_ds_name,
                workspace=self.rg_workspace,
            )

            ds.add_records([record])
            print("Record pushed to argilla")

        return user_query
