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

        if config["use_gpt4"]:
            self.use_gpt4 = True
            self.openai_client = OpenAI()
        else:
            self.use_gpt4 = False
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
        self.rg_ds_name = config.get("argilla_target", {}).get("name", None)
        self.rg_workspace = config.get("argilla_target", {}).get("workspace", None)

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

    def generate_with_gpt4(self, messages, temperature=0.5, stop="\n"):
        response = self.openai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            temperature=0.5,
            stop=stop,
        )
        return response.choices[0].message.content

    def generate_user_query(self, available_functions):
        # Generate user query
        if self.use_gpt4:
            user_query = self.generate_with_gpt4(
                messages=[
                    {
                        "role": "user",
                        "content": format_generate_user_query(
                            available_functions=available_functions
                        ),
                    }
                ]
            )
        else:
            lm = (
                self.llama2_model
                + format_generate_user_query(available_functions=available_functions)
                + gen(
                    max_tokens=100,
                    name="query",
                    stop=None,
                    temperature=1,
                )
            )
            user_query = lm["query"]

        # Create record
        if self.rg:
            print("Creating argilla record for generated user query...")
            record = self.rg.FeedbackRecord(
                fields={
                    "available_functions": available_functions,
                    "user_query": user_query,
                },
            )

            # Push record to argilla dataset
            ds = self.rg.FeedbackDataset.from_argilla(
                name=self.rg_ds_name,
                workspace=self.rg_workspace,
            )

            ds.add_records([record])
            print("Record pushed to argilla")

    def generate_irca_agent_trace(
        self, available_functions, user_query, satisfiable=True
    ):
        # Shuffle the list of functions
        functions = json.loads(available_functions)
        random.shuffle(functions)

        available_functions = json.dumps(functions)

        default_agent_system_message = {
            "role": "system",
            "content": "You are a helpful assistant. Follow user instructions and continue the user message. Do your best to answer user query, using the available functions at your disposal. Create fictitious functions output when required.",
        }

        user_query_prompt = user_query_prompt_template.format(
            available_functions=available_functions,
            does_or_does_not="does" if satisfiable else "does not",
        )

        # print("prompt after instantiation:")
        # print(user_query_prompt)

        workflow = []
        agent_scratchpad = "Thought: "

        # # Generate user query
        # if self.use_gpt4:
        #     user_query = self.generate_with_gpt4(
        #         messages=[
        #             default_agent_system_message,
        #             {
        #                 "role": "user",
        #                 "content": user_query_prompt[
        #                     0 : user_query_prompt.find("\n\nSure !")
        #                 ],
        #             },
        #             {
        #                 "role": "assistant",
        #                 "content": "Sure ! Here is a synthetic query that a user may ask to the AI ASSISTANT: ",
        #             },
        #         ]
        #     )
        # else:
        #     lm = (
        #         self.llama2_model
        #         + user_query_prompt
        #         + gen(
        #             max_tokens=500,
        #             name="query",
        #             stop="\n",
        #             temperature=1,
        #         )
        #     )
        #     user_query = lm["query"]
        workflow.append(("user_query", user_query))

        # Generate agent thought and function calls iteratively
        if self.use_gpt4:
            thought = self.generate_with_gpt4(
                messages=[
                    default_agent_system_message,
                    {
                        "role": "user",
                        "content": agent_prompt_template.format(
                            available_functions=available_functions,
                            user_query=user_query,
                            agent_scratchpad=agent_scratchpad,
                        ),
                    },
                ],
                temperature=0.2,
            )
        else:
            lm_before = self.llama2_model + agent_prompt_template.format(
                available_functions=available_functions,
                user_query=user_query,
                agent_scratchpad="",
            )
            lm += "Thought: " + gen(
                max_tokens=500,
                name="thought",
                stop="\n",
                temperature=0.25,
            )
            thought = lm["thought"]
        workflow.append(
            {
                "type": "thought",
                "lm_before": lm_before,
                "prefix": "Thought: ",
                "thought": thought,
                "suffix": "",
            }
        )
        agent_scratchpad += thought

        if self.use_gpt4:
            action_choice = self.generate_with_gpt4(
                messages=[
                    default_agent_system_message,
                    {
                        "role": "user",
                        "content": agent_prompt_template.format(
                            available_functions=available_functions,
                            user_query=user_query,
                            agent_scratchpad=agent_scratchpad,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "Let's choose one action amongst the following possibles choices: 'call a function', 'ask the user for more information', 'give a final answer', 'abort because I don't have the capabilities to fulfill the user request'].\nAction choice: ",
                    },
                ],
                temperature=0,
            )
        else:
            lm_before = lm
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
            action_choice = temp_lm["action_choice"]
        workflow.append(
            {
                "type": "action_choice",
                "lm_before": lm_before,
                "prefix": " Next, I will ",
                "output": action_choice,
                "suffix": "",
            }
        )

        i = 0
        while (i < 10) and ("call" in action_choice.lower()):
            i += 1

            # info_completeness_score = self.validate_info_completeness(lm)
            # if not info_completeness_score >= 4:
            #     agent_scratchpad += "\nAn internal evaluation revealed that some information is missing, so I need to identify what's lacking and ask the user for specific details. It's important to be clear and precise in my queries to efficiently gather the necessary data for successful task completion."
            #     break  # i need more information

            agent_scratchpad += "\nCall function: "
            if "call" in action_choice.lower():
                if self.use_gpt4:
                    call_function = self.generate_with_gpt4(
                        messages=[
                            default_agent_system_message,
                            {
                                "role": "user",
                                "content": agent_prompt_template.format(
                                    available_functions=available_functions,
                                    user_query=user_query,
                                    agent_scratchpad=agent_scratchpad,
                                ),
                            },
                        ],
                    )
                else:
                    lm += (
                        '\nCall function: {"name": "'
                        + {gen("fct_name", stop='"')}
                        + '"}, "parameters": '
                        + gen(
                            max_tokens=500,
                            name="fct_parameters",
                            stop="\n",
                            temperature=0,
                        )
                    )
                    temp_lm += "All information used to call the function above is available within the context (yes/no): "
                    lm += "\n<wait_for_output>"
                    call_function = lm["call_function"]
                workflow.append(
                    (
                        "call_function",
                        {"name": lm["fct_name"], "parameters": lm["fct_parameters"]},
                    )
                )
                agent_scratchpad += call_function + "\n<wait_for_output>"
                # Generate synthetic function output
                output_shortuuid = shortuuid.uuid()
                agent_scratchpad += f"\nOutput[{output_shortuuid}]: "
                if self.use_gpt4:
                    output_txt = self.generate_with_gpt4(
                        messages=[
                            default_agent_system_message,
                            {
                                "role": "user",
                                "content": agent_prompt_template.format(
                                    available_functions=available_functions,
                                    user_query=user_query,
                                    agent_scratchpad=agent_scratchpad,
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": "Sure ! I don't have access to that function but I will generate a fictitious output that may be returned by such function call, in single line dictionary format:"
                                + "\nOutput[{output_shortuuid}]: ",
                            },
                        ],
                    )
                else:
                    lm += f"\nOutput[{output_shortuuid}]: " + gen(
                        max_tokens=500,
                        name="function_output",
                        stop="\n",
                        temperature=1,
                    )
                    output_txt = lm["function_output"]
                workflow.append(
                    (
                        "function_output",
                        output_shortuuid,
                        output_txt,
                    )
                )
                agent_scratchpad += output_txt

                # Genere new thought
                agent_scratchpad += "\nThought: "
                if self.use_gpt4:
                    thought = self.generate_with_gpt4(
                        messages=[
                            default_agent_system_message,
                            {
                                "role": "user",
                                "content": agent_prompt_template.format(
                                    available_functions=available_functions,
                                    user_query=user_query,
                                    agent_scratchpad=agent_scratchpad,
                                ),
                            },
                        ],
                        temperature=0.2,
                    )
                else:
                    lm += "\nThought: " + gen(
                        max_tokens=500,
                        name="thought",
                        stop="\n",
                        temperature=0.25,
                    )
                    thought = lm["thought"]
                workflow.append(("thought", thought))
                agent_scratchpad += thought

                # Generate new action choice
                if self.use_gpt4:
                    action_choice = self.generate_with_gpt4(
                        messages=[
                            default_agent_system_message,
                            {
                                "role": "user",
                                "content": agent_prompt_template.format(
                                    available_functions=available_functions,
                                    user_query=user_query,
                                    agent_scratchpad=agent_scratchpad,
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": "Let's choose one action amongst the following possibles choices: 'call a function', 'ask the user for more information', 'give a final answer', 'abort because I don't have the capabilities to fulfill the user request'].\nAction choice: ",
                            },
                        ],
                        temperature=0,
                    )
                else:
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
                    action_choice = temp_lm["action_choice"]
                workflow.append(("action_choice", action_choice))

        agent_scratchpad += "\n\n### FINAL ANSWER\n"
        if self.use_gpt4:
            final_answer = self.generate_with_gpt4(
                messages=[
                    default_agent_system_message,
                    {
                        "role": "user",
                        "content": agent_prompt_template.format(
                            available_functions=available_functions,
                            user_query=user_query,
                            agent_scratchpad=agent_scratchpad,
                        ),
                    },
                ],
                temperature=0.2,
                stop=None,
            )
        else:
            lm += "\n\n### FINAL ANSWER\n" + gen(
                max_tokens=500,
                name="final_answer",
                temperature=0.5,
            )
            final_answer = lm["final_answer"]
        workflow.append(("final_answer", final_answer))

        # print(json.dumps(workflow, indent=4))

    def build_agent_trace():
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
        return trace

    def build_argilla_record(self, available_functions, user_query, trace):
        # Create record
        if self.rg:
            print("Creating argilla record for generated user query...")
            record = self.rg.FeedbackRecord(
                fields={
                    "available_functions": available_functions,
                    "user_query": user_query,
                    "agent_trace": trace,
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
    
    def push_record_to_argilla_dataset(self, record):
        # Push record to argilla dataset
        ds = self.rg.FeedbackDataset.from_argilla(
            name=self.rg_ds_name,
            workspace=self.rg_workspace,
        )

        ds.add_records([record])
        print("Record pushed to argilla")

    # def analyze_function_output(lm, output):
    #     """
    #     Analyzes the output of a function execution to determine if it was successful.

    #     :param fct_name: Name of the function to execute.
    #     :param parameters: Parameters for the function call.
    #     :return: Output of the function (could be a success message, error, or data).
    #     """
    #     temp_lm = lm + "\nCalling the function {fct_name} with parameters {parameters} generated the following output: ."
    #     # Implementation of function execution logic goes here
    #     # For now, this is a placeholder returning a success message or error based on function name
    #     if fct_name == "create_contact":
    #         if parameters.get("email") == "alice@example.com":
    #             return "Error: Can't create email 'alice@example.com': email is not available"
    #         return "Success: Contact created"
    #     # Add more conditions for other functions if needed
    #     return "Success"
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

    def function_not_available(self, lm, fct_name):
        