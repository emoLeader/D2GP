# From Demand to Grounded Plan: Task Customization and Planning for Service Robots with Deep Learning and LLMs

A **user-demand-oriented**, **environment-adaptive** task planning system for service robots, integrating:

- **UATC** (User-Centric Adaptive Task Customization)
- **TSGP** (Task-Driven Scene Graph Pruning) 
- **GTP** (Grounded Task Planning)

---

## Execution

### 1. Clone the repository and navigate into it
```bash
git clone https://github.com/emoLeader/D2GP.git
cd D2GP
```
### 2. Create and activate a Conda environment (Linux)
```bash
conda create -n d2gp python=3.12.3 -y
conda activate d2gp
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Enter the src directory
```bash
cd src
```

### 5. Train the demand–task contrastive learning model
```bash
python DT_train.py
```

### 6. Task customization (UATC)
```bash
# Configure user_id, input_text_demand, or input_text_task in UATC.py
python UATC.py
```

### 7. Overall task planning framework (including TSGP)
```bash
python GTP.py
```

### 8. User Preference module evaluation
```bash
cd evaluate_preferences
python evaluate_preferences.py
```

## 📁 Repository Structure

```text
D2GP/
├── json/
│   ├── available_actions.json
│   ├── available_examples.json
│   ├── demand_tasks.json
│   ├── grounded_tasks.json
│   ├── position_relationships_state.json
│   └── all_objects.json
│   └── demands.json
│   └── tasks.json
│   └── user_preferences.json
├── src/
│   ├── DT_train.py
│   ├── DT_test.py
│   ├── UATC.py
│   ├── GTP.py
│   ├── recursive_traversal.py
│   ├── tools/
│   │   ├── fix.py
│   │   ├── generate_tasks_by_objects.py
│   │   └── generate_tasks_for_demands.py
│   └── evaluate_preferences/
│       ├── demand_tasks.json
│       ├── grounded_tasks.json
│       ├── user_preferences.json
│       ├── user_demand_ground_truth.json
│       └── evaluate_preferences.py
├── requirements.txt
├── setup.txt
├── README.md
└── LICENSE
```

