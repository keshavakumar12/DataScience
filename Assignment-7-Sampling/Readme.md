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
```

Key outputs (latest run)
- `accuracy_matrix_final.csv`
- `results_long_final.csv`
- `best_by_model_final.csv`
- `best_by_sampling_final.csv`
