prompt_template = """You are an expert tutor teaching another tutor how not to respond to a student's response.

When revising tutor responses, use the following hierarchy of valuable aspects (most important at the top). A poor response fails higher-priority items before lower ones:
1. Factuality + Non-contradiction + No Nonsense: The response should be factually correct, should not contradict what the student has said, and should not contain irrelevant information.
2. Mistake Identification + Location: The response should identify, either explicitly or implicitly, that there is a mistake in the student's solution.
3. Scaffolding + Actionability: The response should address the misunderstanding step by step, guiding the student with questions or hints instead of directly giving the full solution.
4. Targetedness: The response should address the student's core misconception.
5. Not revealing the final answer: Avoid giving away the final answer unless the context clearly requires it.
6. Clarity + Coherence: The response should be clear, coherent, and connected to the student's latest reasoning.

Below is a conversation between a student and a tutor:
{conversation}

The correct (gold) solution to the task is:
{gold_solution}

The ideal tutor's next response is:
{response}

Your job is to minimally revise the tutor's response so that it clearly fails to align with the hierarchy above.
- Do not rewrite the response completely.
- Keep as much of the original wording as possible.
- Only adjust what is needed to make the response misaligned.

Return your output as valid JSON:
{{
    "thoughts": "Briefly explain how the response was revised to become misaligned.",
    "revised_response": "Your minimally revised response"
}}
"""
