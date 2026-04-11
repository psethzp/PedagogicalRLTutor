prompt_template = """You are an expert evaluator analyzing a tutor's responses in a learning dialogue.

Below is a conversation between a student and a tutor:
{conversation}

The correct (gold) solution to the task is:
{gold_solution}

Your task is to choose which tutor response is better as the next step in the dialogue. Remember: The tutor's goal is to guide the student toward discovering the correct solution, rather than simply revealing the answer.

Option A: {response_a}
Option B: {response_b}

Return your evaluation as a valid JSON object in the following format:
{{
    "thoughts": "Briefly explain your reasoning for choosing one response over the other (or if they are tied).",
    "better_response": "A"  // or "B", or "Tie"
}}
"""
