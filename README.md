# D2GP: From Demand to Grounded Plan

A **user-demand-oriented**, **environment-adaptive** task planning system for service robots, integrating:

- **UATC** (User-Centric Adaptive Task Customization)
- **TSGP** (Task-Driven Scene Graph Pruning) 
- **GTP** (Grounded Task Planning)
  
> **Paper**: _From Demand to Grounded Plan: Task Customization and Planning for Service Robots with Deep Learning and LLMs_

---

## 📁 Repository Structure

D2GP/
├── json/
│   ├── available_actions.json         
│   ├── available_examples.json           
│   ├── demand_tasks.json                
│   ├── grounded_tasks.json       
│   ├── position_relationships_state.json 
│   └── all_objects.json                  
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


