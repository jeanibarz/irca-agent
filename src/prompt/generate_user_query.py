# Notes:
# - available functions must be a single line list of json schemas
# - agent scratchpad must always end with a new line separator (i.e. LLM completion starts at char 0 of a new line)

prompt_template = """\
### INSTRUCTIONS
Your task is to generate a query that the user could make to the AI. You must ensure that the AI {does_or_does_not} have the appropriate functions to fulfill the query.
Be creative. You can be overly verbose or straight to the point, use familiar language or not, you may make some mistakes in punctuations or grammar.
Ensure to output the user query in a single line. Only output the user query, and nothing else. A new line will end the generation.

### FUNCTIONS AVAILABLE
{available_functions}

### USER QUERY
"""
