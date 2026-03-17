import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

if OUTPUT_DIR not in sys.path:
    sys.path.append(OUTPUT_DIR)

from retriever import Retriever
from rag_pipeline import answer_question


def normalize_text(text):
    return text.replace("。", "").replace("，", "").replace(" ", "").strip().lower()


def exact_match(prediction, ground_truth):
    return normalize_text(prediction) == normalize_text(ground_truth)


def keyword_hit(prediction, keywords):
    pred_norm = normalize_text(prediction)
    hit_count = 0

    for kw in keywords:
        kw_norm = normalize_text(kw)
        if kw_norm in pred_norm:
            hit_count += 1

    if len(keywords) == 0:
        return 0.0

    return hit_count / len(keywords)


def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def load_test_data():
    test_path = os.path.join(DATA_DIR, "qa_test.json")
    with open(test_path, "r", encoding="utf-8") as f:
        return json.load(f)

def retrieval_hit(contexts, keywords):

    for ctx in contexts:

        for kw in keywords:

            if kw in ctx:
                return 1

    return 0

def run_evaluation():
    ensure_output_dir()

    test_data = load_test_data()
    retriever = Retriever()

    predictions = []

    exact_match_count = 0
    total_keyword_score = 0.0
    sufficient_count = 0
    insufficient_count = 0
    retrieval_hits = 0

    for sample in test_data:
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        keywords = sample.get("keywords", [])

        result = answer_question(question, retriever)
        gen = result["generation_result"]
        rh = retrieval_hit(result["contexts"], keywords)
        retrieval_hits += rh

        prediction = gen["answer"]
        status = gen["status"]
        evidence = gen["evidence"]

        em = exact_match(prediction, ground_truth)
        kh = keyword_hit(prediction, keywords)

        if em:
            exact_match_count += 1

        total_keyword_score += kh

        if status == "success":
            sufficient_count += 1
        else:
            insufficient_count += 1

        predictions.append({
            "id": sample["id"],
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "status": status,
            "keywords": keywords,
            "keyword_hit_score": round(kh, 4),
            "exact_match": em,
            "evidence": evidence,
            "contexts": result["contexts"]
        })

    total_samples = len(test_data)
    em_score = exact_match_count / total_samples if total_samples > 0 else 0.0
    avg_keyword_score = total_keyword_score / total_samples if total_samples > 0 else 0.0

    predictions_path = os.path.join(OUTPUT_DIR, "predictions.json")
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    report_lines = []
    retrieval_hit_score = retrieval_hits / total_samples
    report_lines.append("==== Evaluation Report ====")
    report_lines.append(f"Total samples: {total_samples}")
    report_lines.append(f"Exact Match: {em_score:.4f}")
    report_lines.append(f"Average Keyword Hit: {avg_keyword_score:.4f}")
    report_lines.append(f"Retrieval Hit: {retrieval_hit_score:.4f}")
    report_lines.append(f"Success count: {sufficient_count}")
    report_lines.append(f"Insufficient count: {insufficient_count}")
    report_lines.append("")

    report_lines.append("==== Per-sample Results ====")
    for item in predictions:
        report_lines.append(f"[ID] {item['id']}")
        report_lines.append(f"Question: {item['question']}")
        report_lines.append(f"Ground Truth: {item['ground_truth']}")
        report_lines.append(f"Prediction: {item['prediction']}")
        report_lines.append(f"Status: {item['status']}")
        report_lines.append(f"Exact Match: {item['exact_match']}")
        report_lines.append(f"Keyword Hit Score: {item['keyword_hit_score']:.4f}")
        report_lines.append(f"Evidence: {item['evidence']}")
        report_lines.append(f"Contexts: {item['contexts']}")
        report_lines.append("-" * 60)

    report_path = os.path.join(OUTPUT_DIR, "eval_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\nPredictions saved to: {predictions_path}")
    print(f"Evaluation report saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()