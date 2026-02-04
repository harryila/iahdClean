# Phase 4: Analysis & Figures

## Purpose
Generate visualizations and answer research questions.

## Outputs

### Figures (`figures/`)
1. **Figure 8 replications** - 24 ablation curves (one per experiment)
2. **Cross-question heatmaps** - Head overlap between question types
3. **Method comparison** - Which heads each method identifies
4. **Summary tables** - Numerical results

### Final Report (`FINDINGS.md`)
Answers to research questions:

1. **Equal Impact?** Do same # of ablated heads affect different questions equally?
2. **Numerical vs Categorical?** Similarities between year/employee and state heads?
3. **Ablation Effectiveness?** Are ablations equally effective across categories?
4. **Cross-Question Patterns?** Patterns within numerical vs categorical questions?

## Usage
```bash
# Generate all figures
python analysis/generate_figures.py

# Generate specific analysis
python analysis/cross_question_analysis.py
python analysis/method_comparison.py
```
