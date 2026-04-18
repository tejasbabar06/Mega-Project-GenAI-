"""
test_all_methods.py
-------------------
Verifies all 7 feature-selection methods work correctly
on the sample dataset end-to-end through the agent pipeline.
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from agents.data_agent      import DataAnalysisAgent
from agents.planning_agent  import FeaturePlanningAgent
from agents.selection_agent import FeatureSelectionAgent

# Load data & run agents 1 and 2 once (shared across all method tests)
df     = pd.read_csv("sample_dataset.csv")
report = DataAnalysisAgent().analyse(df, "defaulted")
plan   = FeaturePlanningAgent().plan(report)

METHODS = [
    "selectkbest",
    "randomforest",
    "lasso",
    "gradientboosting",
    "extratrees",
    "mutualinfo",
    "pca",
]

print(f"{'Method':<20}  {'Selected Features'}")
print("-" * 80)

all_passed = True
for method in METHODS:
    try:
        agent  = FeatureSelectionAgent(k_features=5, method=method)
        result = agent.select(df, plan)
        feats  = result["selected_features"]
        status = "[OK]"
        print(f"[OK]  {method:<20}  {feats}")
    except Exception as e:
        all_passed = False
        print(f"[FAIL] {method:<19}  ERROR: {e}")

print("-" * 80)
if all_passed:
    print(f"\n  ALL {len(METHODS)} METHODS PASSED - OK")
else:
    print("\n  SOME METHODS FAILED - CHECK ABOVE")
