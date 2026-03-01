# src/eval/run_eval.py
# ============================================================================
# Evaluation Runner — Executes the 20-query taxonomy against both pipelines,
# logs results, and exports aggregate metrics + per-query detail to CSV.
#
# Usage:
#   python src/eval/run_eval.py              # live run (needs API key)
#   python src/eval/run_eval.py --replay     # replay from session cache
# ============================================================================

import os
import sys
import json
import csv
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Ensure repo root is on the path so we can import the RAG module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.rag.rag import run_comparative_rag, _load_session_cache, _find_cached_entry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/eval.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 20-Query Evaluation Set
# ---------------------------------------------------------------------------
EVAL_QUERIES = [
    # ---- Direct Queries (10) ----
    {
        "id": "D01",
        "category": "direct",
        "query": "What are the two major types of attacks identified for large language models applied in smart grids, and how are they validated?",
        "target_source": "Li et al.",
    },
    {
        "id": "D02",
        "category": "direct",
        "query": "What specific tasks in power dispatch does the GAIA Large Language Model address, and how was its training dataset constructed to handle these tasks?",
        "target_source": "Cheng et al.",
    },
    {
        "id": "D03",
        "category": "direct",
        "query": "What is the exact four-stage methodology used to synthesize household daily energy data using knowledge distillation from LLMs?",
        "target_source": "Takrouri et al.",
    },
    {
        "id": "D04",
        "category": "direct",
        "query": "What are the two adaptation strategies compared during the development of EnergyGPT, and which base LLM was utilized?",
        "target_source": "Chebbi et al.",
    },
    {
        "id": "D05",
        "category": "direct",
        "query": "According to the comprehensive literature survey by Sarwar et al., what are the primary challenges limiting the practical implementation of LLMs in power systems?",
        "target_source": "Sarwar et al.",
    },
    {
        "id": "D06",
        "category": "direct",
        "query": "What predictive and prescriptive maintenance strategies do LLMs enable for improving overall grid reliability?",
        "target_source": "Survey sources",
    },
    {
        "id": "D07",
        "category": "direct",
        "query": "How does the direct integration of external weather datasets bypass intermediate modeling stages in household energy modeling?",
        "target_source": "Takrouri et al.",
    },
    {
        "id": "D08",
        "category": "direct",
        "query": "What are the specific threat models associated with prompt injection and bad data injection in smart grid LLMs?",
        "target_source": "Li et al.",
    },
    {
        "id": "D09",
        "category": "direct",
        "query": "How does the parameter-efficient LoRA-based variant of EnergyGPT differ from the full-parameter Supervised Fine-Tuning variant in terms of infrastructure requirements?",
        "target_source": "Chebbi et al.",
    },
    {
        "id": "D10",
        "category": "direct",
        "query": "How does the GAIA architecture improve upon traditional methods regarding human-machine collaboration in real-time advanced power dispatch scenarios?",
        "target_source": "Cheng et al.",
    },
    # ---- Synthesis / Multi-hop Queries (5) ----
    {
        "id": "S01",
        "category": "synthesis",
        "query": "Compare the deployment infrastructure challenges discussed in the EnergyGPT paper with the computational cost limitations highlighted in the comprehensive survey by Sarwar et al.. Where do they align?",
        "target_source": "Chebbi + Sarwar",
    },
    {
        "id": "S02",
        "category": "synthesis",
        "query": "How do the cybersecurity threat models identified by Li et al. complicate the autonomous power dispatch operations proposed by the GAIA architecture?",
        "target_source": "Li + Cheng",
    },
    {
        "id": "S03",
        "category": "synthesis",
        "query": "Synthesize the findings from Mirshekali's review and Sarwar's literature survey regarding the use of LLMs for fault diagnosis in power systems. What consensus emerges about their effectiveness?",
        "target_source": "Mirshekali + Sarwar",
    },
    {
        "id": "S04",
        "category": "synthesis",
        "query": "Both Takrouri et al. (Household Energy Modeling) and Cheng et al. (GAIA) focus on specific operational tasks. Compare their differing approaches to handling domain-specific data scarcity.",
        "target_source": "Takrouri + Cheng",
    },
    {
        "id": "S05",
        "category": "synthesis",
        "query": "What are the primary risk factors and threat models associated with deploying LLMs in smart grids?",
        "target_source": "Li + Madani",
    },
    # ---- Edge-Case / Adversarial Queries (5) ----
    {
        "id": "E01",
        "category": "edge_case",
        "query": "Does the corpus contain evidence that Large Language Models can physically actuate robotic arms to replace broken transformers in active distribution networks?",
        "expected": "abstain",
    },
    {
        "id": "E02",
        "category": "edge_case",
        "query": "What are the specific hyperparameter settings (learning rate, batch size, and dropout rate) used to train the GPT-5 model for quantum grid routing?",
        "expected": "abstain",
    },
    {
        "id": "E03",
        "category": "edge_case",
        "query": "According to the corpus, how many human casualties were directly caused by LLM hallucinations in European smart grid operations in the year 2024?",
        "expected": "abstain",
    },
    {
        "id": "E04",
        "category": "edge_case",
        "query": "Does the EnergyGPT paper provide a documented C++ codebase for rewriting legacy SCADA firmwares directly?",
        "expected": "abstain",
    },
    {
        "id": "E05",
        "category": "edge_case",
        "query": "According to the provided texts, what is the exact financial cost in USD of deploying GAIA across the entire Texas ERCOT power grid?",
        "expected": "abstain",
    },
]


# ---------------------------------------------------------------------------
# Abstention detector
# ---------------------------------------------------------------------------
ABSTENTION_PHRASES = [
    "i cannot find evidence",
    "not in the provided context",
    "not supported by",
    "not mentioned in",
    "does not contain",
    "no evidence",
    "not available in the provided",
    "not found in the provided",
    "not present in the provided",
]


def detected_abstention(text: str) -> bool:
    """Returns True if the LLM output contains a refusal / abstention signal."""
    lower = text.lower()
    return any(phrase in lower for phrase in ABSTENTION_PHRASES)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_evaluation(replay: bool = False):
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    results = []
    timestamp = datetime.now().isoformat()

    total = len(EVAL_QUERIES)
    logger.info(f"=== Evaluation Run: {total} queries | replay={replay} ===")

    for i, q in enumerate(EVAL_QUERIES, 1):
        qid = q["id"]
        query = q["query"]
        category = q["category"]
        logger.info(f"[{i}/{total}] ({qid} | {category}) {query[:70]}...")

        try:
            result = run_comparative_rag(query=query, replay=replay)
        except SystemExit:
            logger.error(f"  Pipeline exited on {qid}. Skipping.")
            results.append({
                "query_id": qid,
                "category": category,
                "query": query,
                "status": "FAILED",
                "baseline_output": "",
                "enhanced_output": "",
                "baseline_abstained": False,
                "enhanced_abstained": False,
                "baseline_chunk_ids": [],
                "enhanced_chunk_ids": [],
            })
            continue
        except Exception as e:
            logger.error(f"  Unexpected error on {qid}: {e}")
            results.append({
                "query_id": qid,
                "category": category,
                "query": query,
                "status": "ERROR",
                "baseline_output": str(e),
                "enhanced_output": "",
                "baseline_abstained": False,
                "enhanced_abstained": False,
                "baseline_chunk_ids": [],
                "enhanced_chunk_ids": [],
            })
            continue

        baseline_out = result.get("baseline_rag", {}).get("llm_output", "")
        enhanced_out = result.get("enhanced_rag", {}).get("llm_output", "")

        results.append({
            "query_id": qid,
            "category": category,
            "query": query,
            "status": "OK",
            "baseline_output": baseline_out,
            "enhanced_output": enhanced_out,
            "baseline_abstained": detected_abstention(baseline_out),
            "enhanced_abstained": detected_abstention(enhanced_out),
            "baseline_chunk_ids": result.get("baseline_rag", {}).get("retrieved_chunk_ids", []),
            "enhanced_chunk_ids": result.get("enhanced_rag", {}).get("retrieved_chunk_ids", []),
            "timestamp": result.get("timestamp", ""),
        })
        logger.info(f"  ✓ Done. Baseline abstain={detected_abstention(baseline_out)}, "
                     f"Enhanced abstain={detected_abstention(enhanced_out)}")

        # Proactive rate-limit throttle between queries
        # (synthesis queries make 3 LLM calls: decompose + baseline + enhanced)
        if i < total:
            time.sleep(3.0)

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    ok_results = [r for r in results if r["status"] == "OK"]
    edge_results = [r for r in ok_results if r["category"] == "edge_case"]
    non_edge = [r for r in ok_results if r["category"] != "edge_case"]

    edge_abstain_baseline = sum(1 for r in edge_results if r["baseline_abstained"])
    edge_abstain_enhanced = sum(1 for r in edge_results if r["enhanced_abstained"])

    summary = {
        "run_timestamp": timestamp,
        "total_queries": total,
        "successful": len(ok_results),
        "failed": total - len(ok_results),
        "categories": {
            "direct": len([r for r in ok_results if r["category"] == "direct"]),
            "synthesis": len([r for r in ok_results if r["category"] == "synthesis"]),
            "edge_case": len(edge_results),
        },
        "trust_behavior": {
            "edge_case_abstention_rate_baseline": (
                edge_abstain_baseline / len(edge_results) if edge_results else 0
            ),
            "edge_case_abstention_rate_enhanced": (
                edge_abstain_enhanced / len(edge_results) if edge_results else 0
            ),
        },
        "note": "Groundedness/Citation/Usefulness scores require manual review. "
                "See report/phase2_report.pdf for scored rubric.",
    }

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    # Full JSON log
    eval_log_path = "logs/eval_results.json"
    with open(eval_log_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    logger.info(f"Full evaluation log → {eval_log_path}")

    # CSV export for the report
    csv_path = "outputs/eval_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query_id", "category", "query", "status",
            "baseline_abstained", "enhanced_abstained",
            "baseline_chunk_ids", "enhanced_chunk_ids",
        ])
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in writer.fieldnames}
            # Serialize lists to strings for CSV
            row["baseline_chunk_ids"] = "; ".join(r.get("baseline_chunk_ids", []))
            row["enhanced_chunk_ids"] = "; ".join(r.get("enhanced_chunk_ids", []))
            writer.writerow(row)
    logger.info(f"CSV summary → {csv_path}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Queries run:       {summary['total_queries']}")
    print(f"  Successful:        {summary['successful']}")
    print(f"  Failed:            {summary['failed']}")
    print(f"  Categories:        {summary['categories']}")
    print(f"  Edge-case abstention (baseline): "
          f"{summary['trust_behavior']['edge_case_abstention_rate_baseline']:.0%}")
    print(f"  Edge-case abstention (enhanced): "
          f"{summary['trust_behavior']['edge_case_abstention_rate_enhanced']:.0%}")
    print(f"\n  Full log:  {eval_log_path}")
    print(f"  CSV:       {csv_path}")
    print("=" * 70)

    return summary, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run the 20-query evaluation set.")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Use cached sessions instead of live API calls.",
    )
    args = parser.parse_args()
    run_evaluation(replay=args.replay)


if __name__ == "__main__":
    main()
