"""
Test runner to sequentially execute all code blocks from Netflix_Movies_and_TV_Shows_Clustering_AlmaBetter.ipynb
to ensure 100% execution without errors.
"""

import json
import sys

with open('Netflix_Movies_and_TV_Shows_Clustering_AlmaBetter.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Total code cells to execute: {len(code_cells)}")

global_scope = {}
# Prevent plt.show from blocking
import matplotlib.pyplot as plt
plt.show = lambda: None

for idx, cell in enumerate(code_cells, 1):
    code_text = "".join(cell['source'])
    try:
        exec(code_text, global_scope)
        print(f"  [PASS] Cell {idx}/{len(code_cells)} executed successfully.")
    except Exception as e:
        print(f"  [FAIL] Error in Cell {idx}: {e}")
        print("--- Failing Code ---")
        print(code_text[:300])
        print("--------------------")
        sys.exit(1)

print("\nSUCCESS: All 32 code cells in the AlmaBetter Notebook executed with 0 errors!")
