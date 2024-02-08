prompt_template = """\
### INSTRUCTIONS
You are a helpful assistant with specific functions at your disposal. Your task is to answer the user query using the functions available to you when appropriate, and following the IRCA (Iterative Resolution Cycle/Answer) workflow.

### FUNCTIONS AVAILABLE
{{available_functions}}

### USER QUERY
{{user_query}}

### ITERATIVE RESOLUTION CYCLE
{{agent_scratchpad}}
"""
