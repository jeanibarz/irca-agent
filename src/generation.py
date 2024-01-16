import json
import random

import numpy as np
from openai import OpenAI

from src.function_calling.ds_generation.gpt4_functions_v1 import all_available_functions

MAX_FUNCS = len(all_available_functions)
DEFAULT_WORKSPACE = "function_calling"
DEFAULT_DATASET = "user_query_ds"


class PromptGenerator:
    def __init__(self, config):
        """
        Initialize the PromptGenerator with specified configuration parameters.

        Args:
            config (dict): Configuration parameters including min_funcs, max_funcs, etc.
        """
        # Set default values and update with any provided configuration
        default_config = {
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

        self.min_funcs = config["min_funcs"]
        self.max_funcs = config["max_funcs"]
        self.size = config["size"]
        self.distribution_config = config["distribution"]
        self.rg = config.get("rg", None)

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

    def generate_user_query(self, available_functions, satisfiable=True):
        from prompt.generate_user_query import prompt_template

        client = OpenAI()

        prompt = prompt_template.format(
            available_functions=available_functions,
            does_or_does_not="does" if satisfiable else "does not",
        )

        print("prompt after instantiation:")
        print(prompt)

        response = client.chat.completions.create(
            # model="gpt-3.5-turbo",
            model="gpt-4-0613",
            # prompt=prompt,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            stop=["\n"],
        )
        assert len(response.choices) == 1
        first_choice = response.choices[0]
        assert first_choice.finish_reason == "stop"

        user_query = first_choice.message.content.strip()

        # Create record
        if self.rg:
            print("Creating argilla record for generated user query...")
            record = self.rg.FeedbackRecord(
                fields={
                    "available_functions": available_functions,
                    "original_user_query": user_query,
                },
                responses=[
                    {
                        "values": {
                            "corrected_user_query": {
                                "value": user_query,
                            },
                            "is_request_feasible": {
                                "value": "YES" if satisfiable else "NO",
                            },
                        },
                    },
                ],
            )

            # Push record to argilla dataset
            ds = self.rg.FeedbackDataset.from_argilla(
                name=DEFAULT_DATASET, workspace=DEFAULT_WORKSPACE
            )

            ds.add_records([record])
            print("Record pushed to argilla")

        return user_query
