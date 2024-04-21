from llm_model_loader import *

from function_call_parser import extract_function_calls_from_xml
from functools import partial
from response_generator import generate_response


def main():
    initialize_system()
    model_name = "teknium/OpenHermes-2.5-Mistral-7B"
    tokenizer, model = load_transformer_model(model_name)

    generation_func = partial(generate_response, model=model, tokenizer=tokenizer)

    prompts = [
        "Given the current settings, how would you optimize the system?",
        "Compute the sum of 12.5 and 27.3.", f"""
Given the beam prediction scenario, How would you configure the antennas to optimize coverage and signal quality?
""",
    f"""
Given the power allocation constraints, How would you distribute the power across channels to achieve optimal usage while respecting limits?
""",f"""
Given the bandwidth optimization problem, How would you approach solving this problem to minimize the total energy consumption while allocating 50 of bandwidth fairly among 5 users?
""",f"""
Given the bandwidth optimization problem, How would you approach solving this problem to minimize the total energy consumption while allocating 150 of bandwidth fairly among 15 users?
""",f"""
Given the bandwidth optimization problem, How would you approach solving this problem to minimize the total energy consumption while allocating 200 of bandwidth fairly among 20 users?
"""]

    for prompt in prompts:
        print("\n ******************************** Prompt is ********************************************\n",prompt)
        completion = generation_func(prompt)
        print("\n ******************************** Response is ********************************************\n",completion)
        functions = extract_function_calls_from_xml(completion)
        print("Extracted Functions:", functions or "No functions extracted")

if __name__ == "__main__":
    main()
