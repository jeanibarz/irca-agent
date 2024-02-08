class FunctionsFactory:
    def __init__(self, config):
        self.config = config

    def get_available_functions(self):
        version = self.config.get("functions_version", "v1")
        all_functions = FunctionsFactory.load_function_variants(version=version)
        max_funcs = len(all_functions)
        if "max_funcs" in self.config and self.config["max_funcs"] < max_funcs:
            max_funcs = self.config["max_funcs"]
        return all_functions[:max_funcs]

    @staticmethod
    def load_function_variants(version):
        if version == "v1":
            from dataset_generation.function_variants.gpt4_functions_v1 import functions
        elif version == "v2":
            from dataset_generation.function_variants.gpt4_functions_v2 import functions
        elif version == "glaive_v2":
            from dataset_generation.function_variants.glaive_v2_functions import functions
        else:
            raise ValueError("Unknown version of available functions requested.")

        return functions
