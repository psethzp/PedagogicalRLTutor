prompt_template = """Please read the following tutor-student conversations. After each conversation, you will see two possible tutor responses. Your task is to choose which response seems best. When deciding, put yourself in the student's shoes—which response would you find more helpful and easier to understand?

A strong tutor response should:
* Correctly identify and point out the student's mistake
* Avoid simply giving away the answer, instead encouraging active participation
* Offer clear and relevant guidance (such as hints, explanations, or examples) to help the student understand and correct their mistake
* Provide actionable feedback, making it clear what the student should do next
* Maintain coherence, so that the response logically follows from what the student said

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
    "better_response": "A"  // or "B", or "Tie"
}}
"""
