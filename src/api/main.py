import os

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

from core.trace_generator import GuidedTraceGenerator
from dataset_generation.functions_factory import FunctionsFactory


config = {
    "records_nbr_to_generate": 5,
    "model_name_or_path": "/workspace/models/finetuned_models/Mistral-7B-Instruct-v0.2-with-data-augmentation_irca_agent_v5-6.gguf/checkpoint-122",
    "min_funcs": 1,
    "max_funcs": 20,
    "user_request_satisfiable": True,
    "argilla_source": {
        "name": "irca_user_query_dataset_v4",
        "workspace": "irca_agent",
    },
    "argilla_target": {
        "name": "irca_agent_dataset_v5-5",
        "workspace": "irca_agent",
    },
    "use_gpt4": False,
}

# Initialize the PromptGenerator with the given configuration
trace_generator = GuidedTraceGenerator(config)

available_functions = FunctionsFactory.load_function_variants(version="v1")[0:10]


class CompletionInput(BaseModel):
    prompt: str


current_dir = os.path.dirname(__file__)
router = APIRouter()
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@router.post("/completions")
async def generate_text(input_data: CompletionInput):

    traces = [
        trace_generator.generate_single_trace(
            available_functions=available_functions,
            user_query=input_data.prompt,
        )
    ]

    return {"traces": traces}


app.include_router(router=router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="debug", reload=False)
