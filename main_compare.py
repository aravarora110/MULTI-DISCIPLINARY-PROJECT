"""
main_compare.py
───────────────
Run all 7 pipelines, inject results into the dashboard HTML,
then open it automatically in your browser.

Usage:
    python main_compare.py <path_to_train_FD001.txt>
    python main_compare.py          # looks for train_FD001.txt in same folder
"""

import sys, json, os, time, webbrowser, shutil
1
# ── locate the dataset ────────────────────────────────────────────────────────
"""DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "train_FD001.txt"
)"""

DATA_PATH = "/Users/arav/Downloads/mdp/CMAPSSData/train_FD001.txt"

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_HTML = os.path.join(BASE_DIR, "faultsight_dashboard.html")   # the template
OUTPUT_HTML   = os.path.join(BASE_DIR, "results", "dashboard.html")   # written each run

print("\n" + "█" * 65)
print("  FAULTSIGHT — QUANTUM-INSPIRED FAULT DIAGNOSIS")
print("  NASA CMAPSS Dataset · 7 Pipelines")
print("█" * 65 + "\n")

if not os.path.exists(DATA_PATH):
    print(f"✗  Dataset not found: {DATA_PATH}")
    print("   Usage: python main_compare.py /path/to/train_FD001.txt")
    sys.exit(1)

if not os.path.exists(TEMPLATE_HTML):
    print(f"✗  Dashboard template not found: {TEMPLATE_HTML}")
    print("   Make sure faultsight_dashboard.html is in the same folder.")
    sys.exit(1)

# ── pipeline runner ───────────────────────────────────────────────────────────
def run_pipeline(name, filename, data_path):
    import importlib.util
    path = os.path.join(BASE_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t0 = time.time()
    result = mod.run(data_path)
    result["runtime_seconds"] = round(time.time() - t0, 2)
    print(f"  ⏱  Runtime: {result['runtime_seconds']}s\n")
    return result

PIPELINES = [
    ("pipeline_1", "pipeline_1_kbest_smote_svm.py"),
    ("pipeline_2", "pipeline_2_rfe_weights_xgb.py"),
    ("pipeline_3", "pipeline_3_kbest_adasyn_rf.py"),
    ("pipeline_4", "pipeline_4_rfe_smote_et.py"),
    ("pipeline_5", "pipeline_5_kbest_adasyn_xgb.py"),
    ("pipeline_6", "pipeline_6_rfe_weights_svm.py"),
    ("pipeline_7", "pipeline_7_qsvm.py"),
]

results = []
for name, filename in PIPELINES:
    try:
        r = run_pipeline(name, filename, DATA_PATH)
        results.append(r)
    except Exception as e:
        print(f"  ✗ {name} FAILED: {e}\n")
        results.append({"pipeline": name, "error": str(e), "accuracy": 0, "f1_weighted": 0})

# ── rank pipelines ────────────────────────────────────────────────────────────
classical = [r for r in results if "Pipeline 7" not in r.get("pipeline","") and "error" not in r]
quantum   = next((r for r in results if "Pipeline 7" in r.get("pipeline","")), None)
best      = max(classical, key=lambda r: r["f1_weighted"]) if classical else None
leaderboard = sorted(classical, key=lambda r: r["f1_weighted"], reverse=True)

# ── print leaderboard ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  LEADERBOARD (Classical Pipelines)")
print("=" * 65)
print(f"  {'#':<4}{'Pipeline':<14}{'Acc':>8}{'F1':>8}  Classifier")
print("  " + "-" * 55)
for i, r in enumerate(leaderboard, 1):
    marker = " ★ BEST" if r == best else ""
    print(f"  {i:<4}{r['pipeline']:<14}{r['accuracy']:>8.4f}{r['f1_weighted']:>8.4f}  {r['classifier']}{marker}")

if quantum:
    print(f"\n  ⚛  Pipeline 7 (subset)  acc={quantum['accuracy']:.4f}"
          f"  Δ={quantum.get('quantum_advantage_delta', 0):+.4f} vs classical SVM")

print(f"\n  🏆 Best: {best['pipeline']} — {best['classifier']} (F1={best['f1_weighted']:.4f})" if best else "")

# ── build summary object ──────────────────────────────────────────────────────
summary = {
    "data_path": DATA_PATH,
    "pipelines": results,
    "leaderboard": [
        {
            "rank":             i + 1,
            "pipeline":         r["pipeline"],
            "feature_selection":r.get("feature_selection", ""),
            "imbalance_method": r.get("imbalance_method", ""),
            "classifier":       r.get("classifier", ""),
            "accuracy":         r.get("accuracy", 0),
            "f1_weighted":      r.get("f1_weighted", 0),
            "runtime_seconds":  r.get("runtime_seconds", 0),
            "is_best":          r == best,
        }
        for i, r in enumerate(leaderboard)
    ],
    "best_pipeline": best["pipeline"] if best else None,
    "quantum_pipeline": {
        "accuracy":                       quantum.get("accuracy"),
        "classical_accuracy_same_subset": quantum.get("classical_svm_accuracy_same_subset"),
        "quantum_advantage_delta":        quantum.get("quantum_advantage_delta"),
        "n_qubits":                       quantum.get("n_qubits"),
        "note":                           quantum.get("note"),
    } if quantum else None,
}

# ── inject results into dashboard HTML template ───────────────────────────────
os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

with open(TEMPLATE_HTML, "r", encoding="utf-8") as f:
    html = f.read()

results_json = json.dumps(summary, indent=2)
html = html.replace("__RESULTS_JSON__", results_json)

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✓  Dashboard written → {OUTPUT_HTML}")

# ── open in browser ───────────────────────────────────────────────────────────
url = "file://" + os.path.abspath(OUTPUT_HTML)
print(f"✓  Opening browser → {url}\n")
webbrowser.open(url)
