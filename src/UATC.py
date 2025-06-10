import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from DT_test import find_tasks_for_same_demand, find_tasks_for_input_demand  # Call functions from DT_test.py
import numpy as np

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

class UserPreferenceManager:
    """
    Manage user preferences in UATC:
    - Load/save preference records from/to a JSON file
    - Given a user ID and an input (task or demand), extract existing preferences (if any), with optional semantic matching
    - Depending on input type, call find_tasks_for_same_demand or find_tasks_for_input_demand from DT_test.py
    - Extract a label from candidate task text (fallback to using the task text directly)
    - Rank candidate tasks by preference weights
    - Update preference records based on user selection
    """

    def __init__(
        self,
        pref_filepath: str,
        extract_label_fn: callable,
        embed_fn: Optional[callable] = None,
        demand_match_threshold: float = 0.75,
        top_k: int = 5,
    ):
        """
        Args:
            pref_filepath: Path to store the preference JSON
            extract_label_fn: Function to extract a label from task text, signature label = f(task_text, existing_labels)
            embed_fn: Text embedding function for semantic matching; input str returns np.ndarray. If None, only exact matching is used
            demand_match_threshold: Similarity threshold when embed_fn is provided
            top_k: Number of candidates to return from DT_test.py
        """
        self.pref_filepath = pref_filepath
        self.extract_label_fn = extract_label_fn
        self.embed_fn = embed_fn
        self.demand_match_threshold = demand_match_threshold
        self.top_k = top_k

        self._load_preferences()

    def _load_preferences(self) -> None:
        """Load JSON from pref_filepath into self.users_pref."""
        if os.path.exists(self.pref_filepath):
            with open(self.pref_filepath, "r", encoding="utf-8") as f:
                self.users_pref = json.load(f)
        else:
            self.users_pref = []

    def _save_preferences(self) -> None:
        """Write the in-memory self.users_pref back to the JSON file."""
        with open(self.pref_filepath, "w", encoding="utf-8") as f:
            json.dump(self.users_pref, f, ensure_ascii=False, indent=4)

    def _find_user_entry(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Find the record corresponding to user_id; return None if not found."""
        for user in self.users_pref:
            if user.get("user_id") == user_id:
                return user
        return None

    def _match_demand_entry(
        self, pref_list: List[Dict[str, Any]], demand: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find the entry matching a demand in a user's preferences list:
        1. Prefer exact match (lowercased)
        2. If embed_fn exists, perform semantic matching and return the entry with highest similarity ≥ threshold
        """
        demand_norm = demand.strip().lower()
        # 1. Exact match
        for entry in pref_list:
            if entry.get("demand", "").strip().lower() == demand_norm:
                # Only consider it if preferences is non-empty and sum > 0
                if entry.get("preferences") and sum(list(entry["preferences"][i].values())[0] for i in range(len(entry["preferences"]))) > 0:
                    return entry

        # 2. Semantic match
        if self.embed_fn is not None and pref_list:
            demand_emb = self.embed_fn(demand)
            best_entry = None
            best_score = -1.0
            for entry in pref_list:
                text = entry.get("demand", "")
                try:
                    emb = self.embed_fn(text)
                    sim = cosine_similarity(demand_emb, emb)
                    if sim > best_score:
                        best_score = sim
                        best_entry = entry
                except Exception:
                    continue
            if best_score >= self.demand_match_threshold:
                return best_entry
        return None

    def get_preference_dict(
        self, user_id: str, demand: str
    ) -> Tuple[Dict[str, float], bool]:
        """
        Given user_id and current demand, return (pref_dict, is_new_entry):
        - pref_dict: a dictionary mapping labels to weights. If no entry is found or it's empty, return {}
        - is_new_entry: True indicates there is no valid historical preference record for this demand
        """
        user_entry = self._find_user_entry(user_id)
        if user_entry is None:
            return {}, True

        pref_list = user_entry.get("preferences", [])
        matched_entry = self._match_demand_entry(pref_list, demand)
        if matched_entry is None:
            return {}, True

        # Convert non-empty preferences to a dict
        p_list = matched_entry.get("preferences", [])
        if not p_list:
            return {}, True

        p_dict = {list(item.keys())[0]: list(item.values())[0] for item in p_list}
        total_weight = sum(p_dict.values())
        if total_weight <= 0:
            return {}, True

        return p_dict, False

    def get_candidates(
        self, input_text: str, input_is_task: bool
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Based on the input_is_task flag:
        - True: call find_tasks_for_same_demand(input_text, top_k)
        - False: call find_tasks_for_input_demand(input_text, top_k)
        Returns:
          candidates: List of { "task_id": str, "text": str, "sim_score": float }
          matched_demand: The actual demand string used for retrieval
        """
        if input_is_task:
            top_k_results, matched_demand = find_tasks_for_same_demand(input_text, top_k=self.top_k)
        else:
            top_k_results = find_tasks_for_input_demand(input_text, top_k=self.top_k)
            matched_demand = input_text

        candidates = []
        for task_text, sim_score in top_k_results:
            candidates.append({
                "task_id": task_text,
                "text": task_text,
                "sim_score": sim_score
            })
        return candidates, matched_demand

    def rank_candidates_by_preference(
        self,
        candidates: List[Dict[str, Any]],
        pref_dict: Dict[str, float],
        default_weight: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Sort candidate tasks only based on pref_dict, return [(task_id, pref_score)]:
        - If unable to extract a label, use task_id as label
        - pref_score = pref_dict.get(label, default_weight)
        - tie-breaker: sim_score
        """
        scored = []
        for c in candidates:
            task_id = c["task_id"]
            sim_score = c["sim_score"]
            label = self.extract_label_fn(task_id, list(pref_dict.keys())) or task_id
            pref_score = pref_dict.get(label, default_weight)
            scored.append({
                "task_id": task_id,
                "pref_score": pref_score,
                "sim_score": sim_score
            })

        scored.sort(key=lambda x: (x["pref_score"], x["sim_score"]), reverse=True)
        return [(item["task_id"], item["pref_score"]) for item in scored]
        
    def update_preference(
        self,
        user_id: str,
        demand: str,
        chosen_label: str,
        all_candidate_labels: List[str],
        gamma: float = 0.1
    ) -> None:
        """
        Update user preferences using an exponential smoothing strategy to avoid zeroing out a label in a single candidate set:
        - gamma: smoothing coefficient in [0,1], typically around 0.1 or 0.2.
        Algorithm:
        1. If the demand does not exist, create a new entry with all candidate labels initialized to weight 0, then apply smoothing.
        2. Merge all historical labels and current candidate labels, default weight 0.
        3. For each label l: first apply decay p[l] = p[l] * (1 - gamma).
        4. If l == chosen_label, then apply p[l] += gamma.
        5. Finally normalize so that sum(p.values()) = 1.
        """
        user_entry = self._find_user_entry(user_id)
        if user_entry is None:
            user_entry = {"user_id": user_id, "preferences": []}
            self.users_pref.append(user_entry)

        pref_list = user_entry.setdefault("preferences", [])
        matched_entry = self._match_demand_entry(pref_list, demand)

        if matched_entry is None:
            # Entry doesn't exist: initialize p_dict with all candidate labels weight 0,
            # then apply smoothing for chosen_label, then normalize.
            p_dict = {}
            for lbl in all_candidate_labels:
                p_dict[lbl] = 0.0
            # Apply smoothing update for chosen_label
            p_dict[chosen_label] = p_dict.get(chosen_label, 0.0) * (1 - gamma) + gamma

            # Normalize
            total = sum(p_dict.values())
            if total > 0:
                for lbl in p_dict:
                    p_dict[lbl] /= total

            # Write new entry
            new_entry = {
                "demand": demand,
                "preferences": [{lbl: p_dict[lbl]} for lbl in p_dict],
                "timestamp": datetime.now().strftime("%Y-%m-%d")
            }
            pref_list.append(new_entry)
            self._save_preferences()
            return

        # If matched_entry exists, load historical weights
        p_list = matched_entry.get("preferences", [])
        p_dict = {list(item.keys())[0]: list(item.values())[0] for item in p_list}

        # Merge current candidate labels into p_dict (new labels get default weight 0)
        for lbl in all_candidate_labels:
            if lbl not in p_dict:
                p_dict[lbl] = 0.0

        # Exponential smoothing: apply decay to every label
        for lbl in list(p_dict.keys()):
            p_dict[lbl] = p_dict[lbl] * (1 - gamma)

        # Inject smoothing for chosen_label
        p_dict[chosen_label] = p_dict.get(chosen_label, 0.0) + gamma

        # Normalize: if total ≤ 0, assign equal weights
        total = sum(p_dict.values())
        if total <= 0:
            count = len(p_dict)
            if count > 0:
                equal = 1.0 / count
                for lbl in p_dict:
                    p_dict[lbl] = equal
        else:
            for lbl in p_dict:
                p_dict[lbl] /= total

        # Write back matched_entry
        matched_entry["preferences"] = [{lbl: p_dict[lbl]} for lbl in p_dict]
        matched_entry["timestamp"] = datetime.now().strftime("%Y-%m-%d")
        self._save_preferences()

    def get_ordered_tasks(
        self,
        user_id: str,
        input_text: str,
        input_is_task: bool
    ) -> List[str]:
        """
        Complete pipeline:
        1. Get candidate tasks and matched_demand
        2. If candidates is empty, return an empty list
        3. Extract (pref_dict, is_new)
           - If existing preferences found: sort and prompt for manual selection
           - If new or empty preferences: force user selection and initialize preferences
        4. Return final ordered_tasks
        """
        candidates, matched_demand = self.get_candidates(input_text, input_is_task)

        # Improvement 1: if candidates is empty, return immediately
        if not candidates:
            print(f"Sorry, no candidate tasks found for demand “{matched_demand}”.")
            return []

        pref_dict, is_new = self.get_preference_dict(user_id, matched_demand)

        # Existing valid preferences
        if not is_new and pref_dict:
            ordered = self.rank_candidates_by_preference(candidates, pref_dict, default_weight=0.0)
            ordered_tasks = [tid for tid, _ in ordered]

            # Show and ask if user wants to manually select
            print(f"\nDetected existing preferences for demand “{matched_demand}”. Candidates sorted by preference:")
            for idx, task_id in enumerate(ordered_tasks, start=1):
                print(f"{idx}. {task_id}")

            while True:
                choose = input("Would you like to manually pick a task? (y/n): ").strip().lower()
                if choose in {"y", "n"}:
                    break
                print("Invalid input. Please enter 'y' or 'n'.")

            if choose == "y":
                # Interactively let user select until valid
                chosen_task = None
                while True:
                    sel = input("Enter the task number or full task text you want to execute: ").strip()
                    if sel.isdigit():
                        idx = int(sel)
                        if 1 <= idx <= len(ordered_tasks):
                            chosen_task = ordered_tasks[idx - 1]
                            break
                    else:
                        if sel in ordered_tasks:
                            chosen_task = sel
                            break
                    print("Invalid input. Please re-enter a valid number or full task text.")

                # Move chosen_task to first position
                ordered_tasks.remove(chosen_task)
                ordered_tasks.insert(0, chosen_task)

                # Build all_candidate_labels (extract labels from all candidate tasks)
                all_candidate_labels = []
                for t in ordered_tasks:
                    lbl = self.extract_label_fn(t, [])
                    all_candidate_labels.append(lbl or t)
                all_candidate_labels = list(set(all_candidate_labels))

                chosen_label = self.extract_label_fn(chosen_task, all_candidate_labels) or chosen_task
                # Update preferences
                self.update_preference(
                    user_id=user_id,
                    demand=matched_demand,
                    chosen_label=chosen_label,
                    all_candidate_labels=all_candidate_labels
                )
                return ordered_tasks

            # If user chooses 'n', do not update; return current sorted list
            return ordered_tasks

        # New demand or no existing preferences: force user selection
        print(f"\nNo existing preferences found for demand “{matched_demand}”. Please choose from the following candidates:")
        for idx, c in enumerate(candidates, start=1):
            print(f"{idx}. {c['task_id']}")

        chosen_task = None
        while True:
            sel = input("Enter the task number or full task text you want to execute: ").strip()
            if sel.isdigit():
                idx = int(sel)
                if 1 <= idx <= len(candidates):
                    chosen_task = candidates[idx - 1]["task_id"]
                    break
            else:
                if any(sel == c["task_id"] for c in candidates):
                    chosen_task = sel
                    break
            print("Invalid input. Please re-enter a valid number or full task text.")

        # Move chosen_task to first position
        remaining = [c["task_id"] for c in candidates if c["task_id"] != chosen_task]
        ordered_tasks = [chosen_task] + remaining

        # Build all_candidate_labels (extract labels from all candidate tasks; fallback to task_id)
        all_candidate_labels = []
        for t in ordered_tasks:
            lbl = self.extract_label_fn(t, [])
            all_candidate_labels.append(lbl or t)
        all_candidate_labels = list(set(all_candidate_labels))

        # Update preferences (create new entry or initialize then update)
        chosen_label = self.extract_label_fn(chosen_task, all_candidate_labels) or chosen_task
        self.update_preference(
            user_id=user_id,
            demand=matched_demand,
            chosen_label=chosen_label,
            all_candidate_labels=all_candidate_labels
        )
        return ordered_tasks

###############################################################################
# Below example: how to define extract_label_fn and how to call UserPreferenceManager
###############################################################################
import spacy

# Load the spaCy model once
nlp = spacy.load("en_core_web_sm")

# Common verb prefixes to strip
VERB_PREFIXES = ["heat ", "make ", "serve ", "refill ", "pour ", "grab ", "take ", "warm ", "brew "]

def extract_label_spacy(task_text: str, known_labels: List[str]) -> str:
    """
    Use spaCy to extract the 'object' label from a task; steps:
    0. If task_text contains 'milkshake', return 'milkshake' directly.
    1. Check known_labels first; if any known label is contained in text_lower, return it.
    2. Try noun_chunks (take the first chunk containing a noun); if it starts with a verb prefix, strip it and return.
    3. Try dependency parsing to find dobj (direct object); strip verb prefix if present and return.
    4. If still no result, split task_text and return everything except the first word.
    5. Fallback to returning the entire text in lowercase.
    """
    text_lower = task_text.lower()

    # 0. If it contains 'milkshake', return 'milkshake'
    if "milkshake" in text_lower:
        return "milkshake"

    doc = nlp(task_text)

    # 1. Known labels first
    for lbl in known_labels:
        if lbl.lower() in text_lower:
            return lbl

    # 2. Try noun_chunks (take the first chunk containing a noun)
    for chunk in doc.noun_chunks:
        if any(tok.pos_ in {"NOUN", "PROPN"} for tok in chunk):
            chunk_text = chunk.text.lower()
            # If chunk_text starts with a verb prefix, strip it
            for prefix in VERB_PREFIXES:
                if chunk_text.startswith(prefix):
                    return chunk_text[len(prefix):].strip()
            return chunk_text

    # 3. Dependency parse to find dobj (direct object)
    for token in doc:
        if token.dep_ == "dobj":
            obj_span = doc[token.left_edge.i : token.right_edge.i + 1]
            obj_text = obj_span.text.lower()
            for prefix in VERB_PREFIXES:
                if obj_text.startswith(prefix):
                    return obj_text[len(prefix):].strip()
            return obj_text

    # 4. If still nothing, split and return everything except the first word
    parts = text_lower.split()
    if len(parts) > 1:
        return " ".join(parts[1:]).strip()

    # 5. Fallback to entire text in lowercase
    return text_lower

if __name__ == "__main__":
    # Example: assume preference JSON is stored in the current directory as user_preferences.json
    # embed_fn example: can be replaced with a real embedding function
    def dummy_embed(text: str) -> np.ndarray:
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(768)

    pref_manager = UserPreferenceManager(
        pref_filepath="/D2GP/json/user_preferences.json",
        extract_label_fn=extract_label_spacy,  # Use the new extraction function
        embed_fn=dummy_embed,
        demand_match_threshold=0.75,
        top_k=5
    )

    user_id = "001"

    # Scenario 1: Input is a "demand" text
    input_text_demand = "I need a hot drink."
    ordered_tasks_demand = pref_manager.get_ordered_tasks(user_id, input_text_demand, input_is_task=False)
    print("\nFinal returned candidate tasks order (by preference/after selection):", ordered_tasks_demand)

    # Scenario 2: Input is a "task" text
    # input_text_task = "wash plate"
    # ordered_tasks_task = pref_manager.get_ordered_tasks(user_id, input_text_task, input_is_task=True)
    # print("\nFinal returned candidate tasks order (by preference/after selection):", ordered_tasks_task)