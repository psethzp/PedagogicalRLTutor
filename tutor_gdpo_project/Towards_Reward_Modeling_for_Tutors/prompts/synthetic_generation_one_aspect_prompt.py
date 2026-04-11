prompt_template = """You are an expert tutor helping another tutor refine their response.

Your task is to improve the response on only one criterion:
{aspect}

Below is a conversation between a student and a tutor:
{conversation}

The correct (gold) solution to the task is:
{gold_solution}

The tutor's next response is:
{response}

Your job is to minimally revise the tutor's response so it better aligns with the criterion above.
- Do not rewrite the response completely.
- Keep as much of the original wording as possible.
- Only adjust what is needed to improve alignment with that criterion.

Return your output as valid JSON:
{{
    "thoughts": "Briefly explain how the response was improved for this criterion.",
    "revised_response": "Your minimally revised response"
}}
"""
