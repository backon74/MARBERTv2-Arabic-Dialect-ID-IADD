# Fine-Tuned MARBERTv2 for Arabic Dialect Identification on IADD

Code for the paper **"A Comparative Study of Fine-Tuned MARBERTv2 for Arabic Dialect Identification on the IADD Benchmark"** by Hana Wahban, Hamza Alkhaldi and Maqbool Khan (American University of Bahrain), accepted at the IEEE International Conference on Emerging Trends in Engineering and Computing (ETECOM).

One notebook runs every experiment in the paper on a single Google Colab machine.

> **Status:** the notebook is final. The result files (tables, figures, predictions) will be added here when the full run has finished.

## What the notebook does

`ADI_experiments_lean.ipynb` classifies four Arabic dialects (Egyptian, Gulf, Levantine, Maghrebi) from the [IADD dataset](https://github.com/JihadZa/IADD) and compares:

- three classical models on TF-IDF features: Multinomial Naive Bayes, Logistic Regression, Linear SVM;
- four transformers: MARBERTv2, AraBERT, CAMeLBERT-DA and AraBART;
- four fine-tuning depths of MARBERTv2: linear probe, last 2 layers, last 4 layers, full fine-tuning;
- balanced down-sampling against a larger, imbalanced training set with focal loss.

### Protocol

| Step | What is done |
|---|---|
| Cleaning | Iraqi and "general" samples are excluded, text is normalized, and duplicates are removed before splitting (7,411 repeated copies and 306 sentences with conflicting dialect labels) |
| Split | One fixed split shared by every model: 911 test and 364 validation samples per dialect |
| Training sets | Balanced: 3,280 samples per dialect. Capped: up to 12,000 per dialect |
| Classical tuning | Grid search with 5-fold cross-validation on the training set |
| Transformer tuning | Learning rate chosen on the validation split from 1e-5, 2e-5, 3e-5, 5e-5 (MARBERTv2); the other transformers reuse it |
| Model selection | Best checkpoint by validation macro F1; the test set is used once per run |
| Seeds | 5 for full MARBERTv2 fine-tuning, 3 for the other transformer configurations |
| Robustness | Bootstrap 95% confidence intervals, McNemar's test, paired t-tests, and two extra random splits |
| Cost | Training and inference time, throughput, peak GPU memory, model size, and energy estimated with CodeCarbon |

## How to run

1. Open `ADI_experiments_lean.ipynb` in [Google Colab](https://colab.research.google.com) and choose `Runtime > Change runtime type > T4 GPU`.
2. In Section 2, set `SMOKE_TEST = True` and choose `Runtime > Run all`. This runs everything on tiny subsets in 5 to 10 minutes and checks that all four models download and train.
3. Set `SMOKE_TEST = False` and run all cells again. The full run takes roughly 5 GPU-hours on a T4.
4. If the session disconnects, reconnect and run all cells again. Finished runs are skipped.

The notebook mounts Google Drive and writes to `MyDrive/Dialect-Classification-Arabic/`. Change `PROJECT_DIR` and `CKPT_DIR` in Section 2 to run it elsewhere. IADD is downloaded automatically from its public repository; the dataset is not redistributed here.

## Outputs

| File or folder | Contents |
|---|---|
| `results.jsonl` | One row per run: metrics, timings, hardware, library versions |
| `predictions/` | Test-set predictions of every run |
| `tables/` | The tables behind the paper: overall results, per-dialect results, significance tests, confidence intervals, extra splits, resources |
| `figures/` | Confusion matrices and the other figures |
| `split/manifest.json` | Counts and settings of the fixed split |

## Environment

Tested on Google Colab with a Tesla T4 GPU, Python 3.13, PyTorch 2.11.0, Transformers 5.18.0 and scikit-learn 1.6.1. See `requirements.txt`.

Pretrained checkpoints: `UBC-NLP/MARBERTv2`, `aubmindlab/bert-base-arabertv02`, `CAMeL-Lab/bert-base-arabic-camelbert-da`, `moussaKam/AraBART`.

With Transformers 5, AraBART's tokenizer does not add the `<s>` and `</s>` tokens that its classification head needs. The notebook detects this and adds them.

## Citation

The citation will be added once the proceedings are published.

## License and acknowledgements

The code is released under the MIT License (see `LICENSE`). IADD and the pretrained models are the work of their respective authors and keep their own terms.
