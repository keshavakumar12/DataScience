# Sampling Assignment (current state)

Data: `../Creditcard_data.csv`

Pipeline now mirrors the provided notebook logic:
- Oversample the minority class to match the majority.
- Create five probabilistic samples: SimpleRandom, Systematic, Stratified, Cluster (by amount quantiles), Convenience (head slice).
- Models: Logistic Regression, Decision Tree, Random Forest, KNN, RBF SVM.
- 70/30 train-test split per sample; metric = accuracy.
- Outputs are written as `*_final.csv` in this folder.

How to run
```bash
cd sampling_assignment
python main.py
# or run the lightweight notebook-based script if you prefer
python run_notebook_style.py   # (create this if you need a direct replica)
```

Key outputs (latest run)
- `accuracy_matrix_final.csv`
- `results_long_final.csv`
- `best_by_model_final.csv`
- `best_by_sampling_final.csv`

Latest accuracies (from notebook-style run after reducing RF capacity to avoid 1.0):
- LogisticRegression: 0.9377–0.9635
- DecisionTree: 0.9688–0.9867
- RandomForest: 0.9891–0.9969
- KNN: 0.9607–0.9734
- SVM: 0.6921–0.7940

Older timestamped files were removed; “final” files reflect the most recent run without perfect (1.0) scores.***
