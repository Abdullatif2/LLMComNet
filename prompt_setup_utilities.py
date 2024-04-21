
import json
from langchain.utils.openai_functions import convert_pydantic_to_openai_function



def setup_prompt_with_functions(prompt, *function_classes):
    """
    Prepares a prompt with descriptions of available functions formatted for the model.

    Args:
    - prompt (str): Initial user query or instruction.
    - function_classes (tuple): Pydantic classes representing the functions the model can perform.

    Returns:
    - str: The enhanced prompt including function usage details.
    """
    function_descriptions = "\n".join(json.dumps(convert_pydantic_to_openai_function(cls), indent=2) for cls in function_classes)
    fn_template = '{"name": "function_name", "arguments": {"arg_1": "value_1", "arg_2": "value_2", ...}}'
    full_prompt = f"""system
You are a helpful assistant with access to the following functions:
{function_descriptions}

To use these functions respond with:
    {fn_template}
    {fn_template}
    ...

Edge cases you must handle:
- If there are no functions that match the user request, you will respond politely that you cannot help.
user
{prompt}
assistant"""
    return full_prompt