# D2GP: From Demand to Grounded Plan

A **user-demand-oriented**, **environment-adaptive** task planning system for service robots, integrating:

- **UATC** (User-Centric Adaptive Task Customization)
- **TSGP** (Task-Driven Scene Graph Pruning) 
- **GTP** (Grounded Task Planning)
  
> **Paper**: _From Demand to Grounded Plan: Task Customization and Planning for Service Robots with Deep Learning and LLMs_
> 
---

## 🚀 Features


---

## 📁 Repository Structure

D2GP/
├── json/
│   ├── available_actions.json            # Standard action templates for mapping LLM output to robot actions
│   ├── available_examples.json           # Example tasks + multi-step action sequences (demonstration prompts)
│   ├── demand_tasks.json                 # Demand–task samples for semantic contrastive learning in UATC
│   ├── grounded_tasks.json               # Environment-adaptive grounded tasks for UATC retrieval
│   ├── position_relationships_state.json # Full scene graph (positional & state info) for TSGP pruning
│   └── all_objects.json                  # List of all objects in the environment
│
├── src/
│   ├── DT_train.py                       
│   ├── DT_test.py                       
│   ├── UATC.py                           
│   ├── GTP.py                            
│   ├── recursive_traversal.py            
│   ├── tools/
│   │   └── fix.py
│   │   └── generate_tasks_by_objects.py
│   │   └── generate_tasks_for_demands.py
│   └── evaluate_preferences/             
│       ├── demand_tasks.json
│       ├── grounded_tasks.json
│       ├── user_preferences.json
│       ├── user_demand_ground_truth.json
│       └── evaluate_preferences.py
│
├── requirements.txt
├── setup.txt
├── README.md
└── LICENSE


