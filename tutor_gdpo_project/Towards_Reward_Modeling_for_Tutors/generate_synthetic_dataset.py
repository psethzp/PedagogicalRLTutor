import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from tqdm import tqdm

from prompts.synthetic_generation_one_aspect_prompt import (
    prompt_template as one_aspect_prompt_template,
)
from prompts.synthetic_generation_prompt import prompt_template as all_aspects_prompt_template
from prompts.synthetic_degradation_one_aspect_prompt import (
    prompt_template as one_aspect_degradation_prompt_template,
)
from prompts.synthetic_degradation_prompt import (
    prompt_template as all_aspects_degradation_prompt_template,
)


IDEAL_ANNOTATION: Dict[str, str] = {
    "Revealing_of_the_Answer": "No",
    "Providing_Guidance": "Yes",
    "Actionability": "Yes",
    "Coherence": "Yes",
}

IMPROVE_ASPECTS: Dict[str, str] = {
    "MistakeIdentification": (
        "Mistake Identification + Location: The response should identify, either explicitly "
        "or implicitly, that there is a mistake in the student's solution."
    ),
    "Scaffolding": (
        "Scaffolding + Actionability: The response should address the misunderstanding "
        "step by step, guiding the student with questions or hints instead of directly "
        "giving the full solution."
    ),
    "Targetedness": "Targetedness: The response should address the student's core misconception.",
    "Clarity": (
        "Clarity + Coherence: The response should be clear, coherent, and connected to the "
        "student's latest reasoning."
    ),
}

DEGRADE_ASPECTS: Dict[str, str] = {
    "Factuality": (
        "Factuality + Non-contradiction + No Nonsense: The response should be factually correct, "
        "should not contradict what the student has said, and should not contain irrelevant information."
    ),
    "MistakeIdentification": (
        "Mistake Identification + Location: The response should identify, either explicitly or "
        "implicitly, that there is a mistake in the student's solution."
    ),
    "Scaffolding": (
        "Scaffolding + Actionability: The response should address the misunderstanding step by step, "
        "guiding the student with questions or hints instead of directly giving the full solution."
    ),
    "Targetedness": "Targetedness: The response should address the student's core misconception.",
    "RevealingAnswer": (
        "Not revealing the final answer: While it is sometimes acceptable to share a sub-step answer, "
        "the tutor should avoid giving away the final answer."
    ),
    "Clarity": (
        "Clarity + Coherence: The response should be clear, coherent, and connected to the student's "
        "latest reasoning."
    ),
}


def _prepare_and_literal_eval(answer: str) -> Any:
    answer = answer.replace("```json\n", "")
    answer = answer.replace("\n```", "")
    answer = re.sub(r'(["{}\[\]],?) *\n *(["{}\[\]])', r"\g<1>\g<2>", answer)
    if answer.startswith("{"):
        answer = re.sub(r"}\n+{", "},{", answer)
        answer = re.sub(r"}\n+", "},{", answer)
        answer = f"[{answer}]"
    answer = answer.replace("\n", "\\n")
    answer = re.sub(r"([a-zA-Z])'(s|ve|d|re|m)", r"\g<1>\\'\g<2>", answer)
    if answer.startswith("{"):
        answer = re.sub(r"}\n+{", "}, {", answer)
        answer = re.sub(r"}\n+", "}", answer)
        answer = f"[{answer}]"

    try:
        return ast.literal_eval(answer)
    except Exception:
        return []


def _extract_json_from_text(output: str) -> Any:
    match = re.search(r"```json*\n(.+?)```", output, re.MULTILINE | re.IGNORECASE | re.DOTALL)
    json_block = output
    if match:
        json_block = match.group(1)
    try:
        return json.loads(json_block, strict=False)
    except json.JSONDecodeError:
        try:
            return _prepare_and_literal_eval(json_block)
        except Exception:
            return output


def _init_claude_model(model: str, temperature: float, max_tokens: int):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60,
    )


def _is_not_perfect_response(response: Dict[str, Any], ideal_annotation: Dict[str, str]) -> bool:
    annotation = response.get("annotation", {})
    for key, expected in ideal_annotation.items():
        if annotation.get(key) != expected:
            return True
    return False


def _is_perfect_response(response: Dict[str, Any], ideal_annotation: Dict[str, str]) -> bool:
    return not _is_not_perfect_response(response, ideal_annotation)


def _build_chain(model: str, temperature: float, max_tokens: int, template: str, one_aspect: bool = False) -> Runnable:
    llm = _init_claude_model(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if one_aspect:
        template = PromptTemplate(
            input_variables=["aspect", "conversation", "gold_solution", "response"],
            template=template,
        )
    else:
        template = PromptTemplate(
            input_variables=["conversation", "gold_solution", "response"],
            template=template,
        )
    return template | llm


def _query_revised_response(
    chain: Runnable,
    conversation: str,
    gold_solution: str,
    response: str,
    aspect: str = "",
) -> str:
    payload = {
        "conversation": conversation,
        "gold_solution": gold_solution,
        "response": response,
    }
    if aspect:
        payload["aspect"] = aspect

    response_obj = chain.invoke(payload)
    parsed = _extract_json_from_text(response_obj.content)
    if isinstance(parsed, dict):
        revised = parsed.get("revised_response")
        if isinstance(revised, str) and revised.strip():
            return revised.strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic preference pairs from ranked tutor responses.")
    parser.add_argument("--input-json", required=True, help="Path to ranked response JSON file.")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV file.")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model name.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--max-conversations", type=int, default=None, help="Optional cap for faster experiments.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per response if JSON parsing fails.")
    args = parser.parse_args()

    load_dotenv()

    input_json = Path(args.input_json)
    output_csv = Path(args.output_csv)

    with input_json.open("r", encoding="utf-8") as file_obj:
        data: List[Dict[str, Any]] = json.load(file_obj)

    filtered_data = data
    if args.max_conversations is not None:
        filtered_data = filtered_data[: args.max_conversations]

    improve_all_aspects_chain = _build_chain(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        template=all_aspects_prompt_template,
    )
    improve_one_aspect_chain = _build_chain(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        template=one_aspect_prompt_template,
        one_aspect=True,
    )
    degrade_all_aspects_chain = _build_chain(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        template=all_aspects_degradation_prompt_template,
    )
    degrade_one_aspect_chain = _build_chain(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        template=one_aspect_degradation_prompt_template,
        one_aspect=True,
    )

    pairs: List[Dict[str, Any]] = []
    failed_generations = 0

    for conversation in tqdm(filtered_data, desc="Generating synthetic pairs"):
        conversation_id = conversation.get("conversation_id")
        conversation_history = conversation.get("conversation_history", "")
        gold_solution = conversation.get("ground_truth_solution", "Not Available")
        ranked_responses = conversation.get("ranked_responses", [])

        for ranked_response in ranked_responses:
            original_response = ranked_response.get("response", "")
            if _is_not_perfect_response(ranked_response, IDEAL_ANNOTATION):
                revised_all_aspects = ""
                for _ in range(args.max_retries + 1):
                    revised_all_aspects = _query_revised_response(
                        chain=improve_all_aspects_chain,
                        conversation=conversation_history,
                        gold_solution=gold_solution,
                        response=original_response,
                    )
                    if revised_all_aspects:
                        break
                if revised_all_aspects:
                    pairs.append(
                        {
                            "conversation_id": conversation_id,
                            "conversation_history": conversation_history,
                            "gold_solution": gold_solution,
                            "response_a": original_response,
                            "response_b": revised_all_aspects,
                            "label": 0,
                            "generation_type": "improve",
                            "generation_mode": "all_aspects",
                            "target_aspect": "all_aspects",
                        }
                    )
                else:
                    failed_generations += 1

                for aspect_name, aspect_description in IMPROVE_ASPECTS.items():
                    revised_one_aspect = ""
                    for _ in range(args.max_retries + 1):
                        revised_one_aspect = _query_revised_response(
                            chain=improve_one_aspect_chain,
                            conversation=conversation_history,
                            gold_solution=gold_solution,
                            response=original_response,
                            aspect=aspect_description,
                        )
                        if revised_one_aspect:
                            break
                    if revised_one_aspect:
                        pairs.append(
                            {
                                "conversation_id": conversation_id,
                                "conversation_history": conversation_history,
                                "gold_solution": gold_solution,
                                "response_a": original_response,
                                "response_b": revised_one_aspect,
                                "label": 0,
                                "generation_type": "improve",
                                "generation_mode": "one_aspect",
                                "target_aspect": aspect_name,
                            }
                        )
                    else:
                        failed_generations += 1

            if _is_perfect_response(ranked_response, IDEAL_ANNOTATION):
                degraded_all_aspects = ""
                for _ in range(args.max_retries + 1):
                    degraded_all_aspects = _query_revised_response(
                        chain=degrade_all_aspects_chain,
                        conversation=conversation_history,
                        gold_solution=gold_solution,
                        response=original_response,
                    )
                    if degraded_all_aspects:
                        break
                if degraded_all_aspects:
                    pairs.append(
                        {
                            "conversation_id": conversation_id,
                            "conversation_history": conversation_history,
                            "gold_solution": gold_solution,
                            "response_a": original_response,
                            "response_b": degraded_all_aspects,
                            "label": 1,
                            "generation_type": "degrade",
                            "generation_mode": "all_aspects",
                            "target_aspect": "all_aspects",
                        }
                    )
                else:
                    failed_generations += 1

                for aspect_name, aspect_description in DEGRADE_ASPECTS.items():
                    degraded_one_aspect = ""
                    for _ in range(args.max_retries + 1):
                        degraded_one_aspect = _query_revised_response(
                            chain=degrade_one_aspect_chain,
                            conversation=conversation_history,
                            gold_solution=gold_solution,
                            response=original_response,
                            aspect=aspect_description,
                        )
                        if degraded_one_aspect:
                            break
                    if degraded_one_aspect:
                        pairs.append(
                            {
                                "conversation_id": conversation_id,
                                "conversation_history": conversation_history,
                                "gold_solution": gold_solution,
                                "response_a": original_response,
                                "response_b": degraded_one_aspect,
                                "label": 1,
                                "generation_type": "degrade",
                                "generation_mode": "one_aspect",
                                "target_aspect": aspect_name,
                            }
                        )
                    else:
                        failed_generations += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(pairs).to_csv(output_csv, index=False)

    print(f"Saved {len(pairs)} synthetic pairs to: {output_csv}")
    if failed_generations:
        print(f"Skipped {failed_generations} examples due to parsing/LLM failures.")


if __name__ == "__main__":
    main()
