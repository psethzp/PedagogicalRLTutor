import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ANNOTATION_SCORES: Dict[str, Dict[str, float]] = {
    "Mistake_Identification": {
        "Yes": 1,
        "To some extent": 0.5,
        "No": 0,
    },
    "Mistake_Location": {
        "Yes": 1,
        "To some extent": 0.5,
        "No": 0,
    },
    "Revealing_of_the_Answer": {
        "Yes (and the answer is correct)": 0.5,
        "Yes (but the answer is incorrect)": 0,
        "No": 1,
    },
    "Providing_Guidance": {
        "Yes": 1,
        "To some extent": 0.5,
        "No": 0,
    },
    "Actionability": {
        "Yes": 1,
        "To some extent": 0.5,
        "No": 0,
    },
    "humanlikeness": {
        "Yes": 1,
        "To some extent": 0.5,
        "No": 0,
    },
    "Coherence": {
        "Yes": 1,
        "To some extent": 0.5,
        "No": 0,
    },
    "Tutor_Tone": {
        "Neutral": 0.5,
        "Encouraging": 1,
        "Offensive": 0,
    },
}

DIMENSION_WEIGHTS: Dict[str, float] = {
    "Mistake_Identification": 0.5,
    "Mistake_Location": 1.0,
    "Revealing_of_the_Answer": 0.25,
    "Providing_Guidance": 1.0,
    "Actionability": 0.5,
    "humanlikeness": 0.25,
    "Coherence": 1.0,
    "Tutor_Tone": 0.05,
}


def calculate_response_score(
    annotation: Dict[str, str],
    annotation_scores: Dict[str, Dict[str, float]],
    dimension_weights: Dict[str, float],
) -> float:
    total_score = 0.0
    for dimension, value in annotation.items():
        if dimension in annotation_scores:
            base_score = annotation_scores[dimension].get(value, 0)
            weight = dimension_weights.get(dimension, 1.0)
            total_score += base_score * weight
    return total_score


def rank_responses(
    data_instance: Dict[str, Any],
    annotation_scores: Dict[str, Dict[str, float]],
    dimension_weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    response_scores: List[Dict[str, Any]] = []
    for model_name, response_data in data_instance.get("anno_llm_responses", {}).items():
        annotation = response_data.get("annotation", {})
        score = calculate_response_score(annotation, annotation_scores, dimension_weights)
        response_scores.append(
            {
                "model": model_name,
                "response": response_data.get("response", ""),
                "annotation": annotation,
                "score": score,
            }
        )
    response_scores.sort(key=lambda x: x["score"], reverse=True)
    return response_scores


def process_all_data(
    data: List[Dict[str, Any]],
    annotation_scores: Dict[str, Dict[str, float]],
    dimension_weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    ranked_data: List[Dict[str, Any]] = []
    for instance in data:
        ranked_responses = rank_responses(instance, annotation_scores, dimension_weights)
        ranked_instance = {
            "conversation_id": instance.get("conversation_id"),
            "conversation_history": instance.get("conversation_history", ""),
            "ground_truth_solution": instance.get("Ground_Truth_Solution", "Not Available"),
            "ranked_responses": ranked_responses,
        }
        ranked_data.append(ranked_instance)
    return ranked_data


def print_model_stats(ranked_data: List[Dict[str, Any]]) -> None:
    model_scores: Dict[str, float] = {}
    model_counts: Dict[str, int] = {}

    for instance in ranked_data:
        for response in instance.get("ranked_responses", []):
            model = response["model"]
            score = response["score"]
            model_scores[model] = model_scores.get(model, 0.0) + score
            model_counts[model] = model_counts.get(model, 0) + 1

    avg_scores = [(model, model_scores[model] / model_counts[model]) for model in model_scores]
    avg_scores.sort(key=lambda x: x[1], reverse=True)

    print("\nAverage Scores by Model:")
    for rank, (model, avg_score) in enumerate(avg_scores, 1):
        print(f"  {rank}. {model}: {avg_score:.2f} (across {model_counts[model]} instances)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MRBench_V2_Ranked.json from MRBench_V2.json.")
    parser.add_argument("--input-json", default="MRBench_V2.json", help="Path to MRBench_V2.json.")
    parser.add_argument("--output-json", default="MRBench_V2_Ranked.json", help="Path for ranked output JSON.")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    with input_path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    ranked_data = process_all_data(data, ANNOTATION_SCORES, DIMENSION_WEIGHTS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(ranked_data, file_obj, indent=2, ensure_ascii=False)

    print(f"Ranked results saved to {output_path}")
    print(f"Total instances processed: {len(ranked_data)}")
    if ranked_data:
        print(f"Responses per instance (first item): {len(ranked_data[0].get('ranked_responses', []))}")
    print_model_stats(ranked_data)


if __name__ == "__main__":
    main()
