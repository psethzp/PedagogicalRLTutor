prompt_template = """Please read the following tutor-student conversations. After each conversation, you will see two possible tutor responses. Your task is to choose which response seems best. When deciding, put yourself in the student's shoes—which response would you find more helpful and easier to understand?

Hierarchy of valuable aspects of tutors' responses (with importance decreasing from top to bottom)
1. Factuality + Non-contradiction + No Nonsense: The response should be factually correct, should not contradict what the student has said, and should not contain irrelevant information. We can use "Both are equally bad" only when this aspect is absent. Example:
Conversation history: [the task about gift cards]
Option A (bad response): If Ms. Jones received 5 gift cards worth $10 each, then 1/3 of the thank you cards contained a gift card, meaning 3 out of every 5 cards had a gift card, and you can use this information to solve for the total number of students in her class. --> Factually incorrect
Option B (better response): That's correct! Now, if 1/3 of the thank you cards contained a gift card, how many thank you cards did she receive in total?

2. Mistake Identification + Location: The response should identify, either explicitly or implicitly, that there is a mistake in the student's solution. For example, saying "Nice try" would miss this aspect. Example:
Conversation history: Student: yay
Tutor: Next, we have to find the value of 6 times 9 + 2. Tutor: What is the value of 6 times 9 + 2?
Student: 99
Option A (better response): That's a great start! Remember, we need to follow the order of operations (PEMDAS). Let's multiply 6 times 9 first, then add 2.
Option B (bad response): Very good try. --> Mistake Identification + Location aspect is missing

3. Scaffolding + Actionability: The response should attempt to address the misunderstanding or problem step by step, guiding the student toward understanding by asking questions or providing hints instead of directly giving the solution. However, if the context shows that scaffolding has not worked, providing the solution is preferable. Example:
Conversation history: Student: yay
Tutor: Next, we have to find the value of 6 times 9 + 2. Tutor: What is the value of 6 times 9 + 2?
Student: 99
Option A: Good try, but remember the order of operations: multiplication should be done before addition. So it should be 6 times 9, then add 2. Let's try again!
Option B (better response): Great try! What was your first step? --> Scaffolding

4. Targetedness: The response should address the core misconception/misunderstanding of a student. Example:
Conversation history: [the task about eggs]
Option A: Ok, remember there are only 12 eggs to begin with --> doesn't point
to any misconception
Option B (better response): It looks like you've got a good start, but let's revisit the single yolks calculation - since 5 eggs had double yolks, that means there are actually 12 - 5 = 7 eggs with single yolks, not 12.

5. Not revealing the final answer: While it is sometimes necessary and acceptable to share the answer to a substep, the tutor should avoid giving away the final answer. Example:
Conversation history: [the task about average speed]
Option A: You're close, but let's re-examine this - if Jason already drove 30 minutes at 60 miles per hour, that means he covered 30 miles, and he still has 90 miles left, so to cover those 90 miles in 1 hour (60 minutes), not 1 hour 30 minutes, he needs to average a speed of 90/1 = 90 miles per hour. --> Reveals the final answer
Option B (better response): You're very close, but remember the question asks for the average speed needed for the *remaining* portion of the drive, not the overall trip.

6. Clarity + Coherence: The tutor's response should be free of awkward, confusing or misleading wording. A good tutor response acknowledges the student's input and connects it to the next step. Example:
Conversation history: [the task about gift cards]
Student: She got 5 gift cards since $50 divided by $10 is equal to 5.
Option A: That's correct! Now, if 1/3 of the thank you cards contained a gift card, how many thank you cards did she receive in total?
Option B (better response): Ok. And if she got 5 thank you cards that had gift cards in them, and these were 1/3 of the total number of thank you card, she got, how many thank you cards did she get total? --> More explicit connection between the student's input and the next step

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
