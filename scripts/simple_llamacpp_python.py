from llama_cpp import Llama

llama_config = dict(
    model_path="/workspace/models/TheBloke/Mistral-7B-Instruct-v0.2-DARE-GGUF/mistral-7b-instruct-v0.2-dare.Q3_K_L.gguf",
    n_ctx=4096,
    n_gpu_layers=33,
    device_map={"": 0},
)

llm = Llama(**llama_config)
output = llm(
    "Q: Give very lengthy and detailed explanation of nuclear fusion. A: ",  # Prompt
    max_tokens=1024,  # Generate up to 32 tokens
    stop=[
        "Q:",
    ],  # Stop generating just before the model would generate a new question
    echo=True,  # Echo the prompt back in the output
)  # Generate a completion, can also call create_completion
print(output)
