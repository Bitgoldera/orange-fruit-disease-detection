# CitrusCare: model results page and safe cleanup

This update is designed to be extracted into the root of the existing `orange_disease_efficientnet` project. It does not include the trained `.keras` models or the image datasets. Keep the versions already on your computer.

## What the results page displays

- Training and validation accuracy and loss using `reports/finetune_history.csv`.
- Best validation accuracy and lowest validation loss from that saved history.
- Final test accuracy, test loss, class-wise precision/recall/F1, and the confusion matrix.
- Batch sizes and other model settings read from the project configuration and scripts.
- Optional orange-input-gate metrics from `reports/orange_gate_test_metrics.json`.

The main disease-classifier history has 15 fine-tuning epochs. It is a fine-tuning history; the ZIP does not contain a separate baseline epoch history. The baseline training code uses the configured batch size of 32. `src/finetune.py` caps the disease-model fine-tuning batch size at 16. `src/train_orange_gate.py` also uses batch size 16.

The current test report records test loss 0.106789 and test accuracy 0.959596 (95 correct predictions out of 99). We do not plot a test curve across epochs: the held-out test set should be used for final evaluation, not for repeatedly choosing the model. The dashboard therefore shows final test metrics and the confusion matrix instead.

## Add the page

1. Extract this update ZIP into the root of `~/orange_disease_efficientnet`, preserving folders. The new page must be at `pages/02_Model_Results.py` and the cleanup tool at `src/cleanup_project.py`.
2. Ensure `pandas`, `matplotlib`, `PyYAML`, `streamlit` and the other requirements are installed in the project's `.venv`.
3. Start the main app from Ubuntu/WSL:

   ```bash
   cd ~/orange_disease_efficientnet
   source .venv/bin/activate
   streamlit run app.py --server.port 8501
   ```

4. Use Streamlit's page navigation to open **02 Model Results**. Its data comes from the `reports/` files and `config.yaml` already in the project.

## Batch sizes found in the actual code

- Baseline classifier: `config.yaml` sets `batch_size: 32`, and `src/train.py` uses that setting.
- Classifier fine-tuning: `src/finetune.py` defines `BATCH_SIZE = min(cfg["batch_size"], 16)`, so the effective batch size is 16.
- Orange/non-orange gate: `src/train_orange_gate.py` sets `BATCH_SIZE = 16`.
- Input image size: 224 × 224 pixels.

## Safe cleanup workflow

Run a preview first:

```bash
python src/cleanup_project.py
```

Review the displayed list. The script is deliberately non-destructive: by default it only prints a proposed plan. If it is appropriate, archive the listed optional variants and experimental reports:

```bash
python src/cleanup_project.py --apply
```

Files are moved into a timestamped `archive/cleanup_.../` folder; they are not deleted. The script keeps `app.py`, `app_multilingual.py`, `app_treatment_report.py`, production model files, source files required for normal training/inference, data folders, configuration, and core result reports. It archives known stale app copies and pilot/augmentation/melanose experiments plus large auxiliary audit reports.

## Important evaluation note

The original classifier's recorded test accuracy is 95.96% on only 99 images. This is promising for a prototype, but the small test set and real-world orange-gate false acceptance/rejection cases mean this is not a validated agricultural diagnostic system. Scores are model outputs, not calibrated probabilities.
