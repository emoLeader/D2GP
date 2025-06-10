import sys
import os
import json
import numpy as np
from typing import List, Dict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))         
SRC_DIR  = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from UATC import UserPreferenceManager
from DT_test import find_tasks_for_same_demand, find_tasks_for_input_demand

def load_ground_truth(path: str) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# w/o User Preference
def get_candidate_list_baseline(pref_manager: UserPreferenceManager,
                                demand: str,
                                top_k: int) -> List[str]:
    candidates, _ = pref_manager.get_candidates(demand, input_is_task=False)
    candidates.sort(key=lambda x: x["sim_score"], reverse=True)
    return [entry["task_id"] for entry in candidates[:top_k]]

# w/ User Preference
def get_candidate_list_with_history(pref_manager: UserPreferenceManager,
                                    user_id: str,
                                    demand: str,
                                    top_k: int,
                                    alpha: float = 0.5) -> List[str]:
    pref_dict, is_new = pref_manager.get_preference_dict(user_id, demand)
    candidates, _ = pref_manager.get_candidates(demand, input_is_task=False)

    merged_list = []
    for entry in candidates:
        task_id    = entry["task_id"]
        sim_score  = entry["sim_score"]
        pref_weight = pref_dict.get(task_id, 0.0)
        combined    = alpha * pref_weight + (1.0 - alpha) * sim_score
        merged_list.append((task_id, combined))

    merged_list.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in merged_list[:top_k]]

# Evaluate Hit@K & MRR
def evaluate_hits_mrr(ground_truth: List[Dict[str, str]],
                      pref_manager: UserPreferenceManager,
                      top_k_values: List[int],
                      mode: str = "baseline",
                      alpha: float = 0.5) -> Dict[str, float]:
    hit_counters = {k: 0 for k in top_k_values}
    rr_sum = 0.0
    n = len(ground_truth)

    for record in ground_truth:
        uid        = record["user_id"]
        demand     = record["demand"]
        true_task  = record["ground_truth_task"]
        max_k      = max(top_k_values)

        if mode == "baseline":
            candidate_list = get_candidate_list_baseline(pref_manager, demand, max_k)
        elif mode == "history":
            candidate_list = get_candidate_list_with_history(pref_manager, uid, demand, max_k, alpha=alpha)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for K in top_k_values:
            if true_task in candidate_list[:K]:
                hit_counters[K] += 1

        if true_task in candidate_list:
            rank = candidate_list.index(true_task) + 1
            rr_sum += 1.0 / rank

    results = {}
    for K in top_k_values:
        results[f"Hit@{K}"] = hit_counters[K] / n
    results["MRR"] = rr_sum / n
    return results

if __name__ == "__main__":
    PREF_FILE = os.path.join(THIS_DIR, "user_preferences.json")
    GT_FILE   = os.path.join(THIS_DIR, "user_demand_ground_truth.json")

    # Use full task IDs, no additional label extraction
    pref_manager = UserPreferenceManager(
        pref_filepath          = PREF_FILE,
        extract_label_fn       = lambda task_text, known: task_text,
        embed_fn               = None,
        demand_match_threshold = 0.75,
        top_k                  = 10 
    )

    ground_truth = load_ground_truth(GT_FILE)
    top_k_values = [1, 3, 5]

    # 1) w/o User Preference
    metrics_baseline = evaluate_hits_mrr(
        ground_truth, pref_manager, top_k_values, mode="baseline"
    )

    # 2) w/ User Preference
    metrics_history = evaluate_hits_mrr(
        ground_truth, pref_manager, top_k_values, mode="history", alpha=0.5
    )

    print("=== Evaluation of User Preference ===")
    print("w/o User Preference")
    for K in top_k_values:
        print(f"Hit@{K}: {metrics_baseline[f'Hit@{K}']:.4f}")
    print(f"MRR: {metrics_baseline['MRR']:.4f}\n")

    print("w/ User Preference")
    for K in top_k_values:
        print(f"Hit@{K}: {metrics_history[f'Hit@{K}']:.4f}")
    print(f"MRR: {metrics_history['MRR']:.4f}")


# === Evaluation of User Preference ===
# w/o User Preference
# Hit@1: 0.1100
# Hit@3: 0.2800
# Hit@5: 0.3900
# MRR: 0.2100

# w/ User Preference
# Hit@1: 0.4400
# Hit@3: 0.5900
# Hit@5: 0.6000
# MRR: 0.5158
