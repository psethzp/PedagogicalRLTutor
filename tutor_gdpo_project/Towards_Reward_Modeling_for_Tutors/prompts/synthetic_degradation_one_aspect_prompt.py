prompt_template = """You are an expert tutor teaching another tutor how not to respond to a student's response.

When revising tutor responses, only focus on the following aspect:
{aspect}

Below is a conversation between a student and a tutor:
{conversation}

The correct (gold) solution to the task is:
{gold_solution}

The ideal tutor's next response is:
{response}

Your job is to minimally revise the tutor's response so that it clearly fails to align with the criterion above.
- Do not rewrite the response completely.
- Keep as much of the original wording as possible.
- Only adjust what is needed to make the response misaligned.

Return your output as valid JSON:
{{
    "thoughts": "Briefly explain how the response was revised to become misaligned on this criterion.",
    "revised_response": "Your minimally revised response"
}}
"""
