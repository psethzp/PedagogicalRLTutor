prompt_template = """Please read the following tutor-student conversations. After each conversation, you will see two possible tutor responses. Your task is to choose which response seems best. When deciding, put yourself in the student's shoes—which response would you find more helpful and easier to understand?

Checklist you should follow when making a decisison:
1. Is a response factually incorrect, contradicting, or Nonsense? If yes, it should never be preferred. If no, move to the next point.
2. Does response identify, either explicitly or implicitly, that there is a mistake in the student's solution? If no, it shouldn't be preferred. If yes, move to the next point.
3. Does the response attempt to address the misunderstanding or problem step by step? If yes, a more targeted, step-by-step response is preferred. If no, move to the next point.
4. Does one response more directly addresses the core misunderstanding? If yes, this response should be preferred. If no, move to the next point.
5. Does one response ask a question that the student must answer to understand something, while the other response provides this guidance directly? If yes, a response that questions the student is preferred. If no, move to the next point.
6. Does one response reveal the final answer, while the other does not? If yes, response that doesn't reveal the answer is preferred. If no, move to the next point.
7. Do both responses reveal the final answer? If yes, a response that offers clearer and more helpful guidance is preferred. If no, move to the next point.
8. Do two responses address the student's misunderstanding in different but equally effective ways? If yes, they can be considered equally good as long as they provide helpful feedback; otherwise, try to choose the more targeted, coherent, and clear response. If no, move to the next point.
9. Are responses very similar in terms of guidance they give? If yes, try to choose the more coherent and clear response. If you can't, they can be considered equally good

Conversation:
{conversation}

The correct (gold) solution to the task is:
{gold_solution}

From the following teacher responses, select the one that you consider to be best in this context.

Option A: {response_a}
Option B: {response_b}

Return your evaluation as a valid JSON object in the following format:
{{
    "thoughts": "Briefly explain your reasoning for choosing one response over the other (or if they are tied).",
    "better_response": "A"  // or "B", or "Both are equally bad", or "Both are equally good"
}}
"""
