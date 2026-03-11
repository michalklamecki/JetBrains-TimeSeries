# TODO — Agent tasks

1. Implement pipeline to generate windows and labels from all train files:
   - Parse `labeled_anomalies.csv` into per-channel anomaly ranges (use `ast.literal_eval`).
   - For each training file, generate past windows of size `W` ending at `t_end` and labels for horizon `H` (no leakage).
   - Save per-file outputs: either extracted features (Parquet/npz) or compressed raw windows for deep models.

2. Build and run feature-extraction pipeline (use existing `windows_to_feature_df` as baseline):
   - Aggregate features across files into a single dataset (e.g., `artifacts/features.parquet`).
   - Record `starts`, `chan_id` and `file` for traceability.

3. Implement scaling and train/validation split:
   - Fit `StandardScaler` on training features only; transform val/test.
   - Use time-aware split within training channels or leave-channels-out, depending on desired generalization.

4. Train baseline classifier and save results:
   - Train RandomForest and/or LightGBM on features.
   - Handle class imbalance (`class_weight='balanced'` or resampling).
   - Save best model and evaluation metrics to `artifacts/`.

5. Prepare deep-learning pipeline (optional, after baseline):
   - Implement generator using `tf.data.Dataset` or PyTorch `DataLoader` with `mmap`/lazy loading.
   - Train 1D-CNN and/or LSTM on raw windows; compare to baseline.

6. Evaluation and analysis:
   - Compute: precision, recall, F1, PR-AUC, detection latency, false alerts per 1000 steps.
   - Produce plots: precision-recall curve, sample signals with predicted vs true anomaly ranges.

7. Documentation and deliverables:
   - Update `notebooks/solution.ipynb` with pipeline usage examples.
   - Add short `README.md` with run instructions and default parameters.

Notes and defaults:
- Default parameters to try: `W` in {50,100,200}, `H` in {1,5,10,50}, `step` in {1,5,10}.
- Prefer `float32` for stored windows/features to save memory; use `mmap_mode='r'` for large arrays.
- For 16 GB RAM: load raw `.npy` files into memory (they are small) but avoid stacking all windows—generate features per-file or use generators.

Owner: AI assistant — next action: implement pipeline to aggregate windows and extract features (confirm to proceed).