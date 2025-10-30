# IBD-Information-Retrieval-NLP
Natural Language Processing pipeline for information retrieval on Inflammatory Bowel Diseases (IBD) using the PubMed 200k RCT dataset. Includes preprocessing, statistical and deep learning models (TF-IDF + SVM, Transformer embeddings), evaluation and visualization for biomedical text classification and analysis.
Table of contents
•	Project overview
•	Dataset
•	Repository structure
•	Quickstart
•	Reproducible pipelines
•	Results
•	Roadmap
•	Cite / Thesis
•	License
•	Acknowledgments

## Project overview

The goal is to retrieve and classify biomedical text relevant to IBD and biological/therapeutic factors (e.g., anti-TNF) from research abstracts.
We:
1.	Parse and structure abstracts with line-level labels
2.	Explore label balance and text statistics
3.	Train baselines with TF-IDF + SVM
4.	Train neural models with pretrained sentence embeddings (Universal Sentence Encoder)
5.	Evaluate with accuracy/precision/recall, learning curves and confusion matrices
Why this matters: IBD literature is large and heterogeneous.
A clean IR+NLP pipeline helps surface clinically relevant evidence faster.

## Dataset

•	PubMed 200k RCT (train/dev/test files with section-level labels).
•	Not included in the repo. Please download from its official source and place as:

data/
├── train.txt
├── dev.txt
└── test.txt
Optionally you may keep a small sample in data/sample/ for quick runs.

## Repository structure

This project follows a modular structure typical for machine learning and NLP pipelines:
.
├── data/                     # <- place dataset files here (not tracked)
├── notebooks/
│   └── Thesis.ipynb          # exploratory analysis & figures
├── src/
│   ├── preprocessing.py      # text loading, parsing, cleaning
│   ├── svm_model.py          # TF-IDF + SVM pipeline + grid search
│   ├── use_model.py          # Sentence embeddings (USE) + dense head
│   └── utils.py              # metrics, plots, saving
├── results/
│   ├── confusion_matrices/
│   └── learning_curves/
├── requirements.txt
├── LICENSE
└── README.md


Quickstart
1) Create environment
   
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

2) Prepare data
•	Download PubMed 200k RCT and place train.txt, dev.txt, test.txt under data/.

3) Run (current state)
•	Open notebooks/Thesis.ipynb and run all cells
or
•	Run the SVM pipeline (after refactor)
python src/svm_model.py --data_dir data --out_dir results

•	Run the sentence-embeddings pipeline (after refactor)
python src/use_model.py --data_dir data --out_dir results



Reproducible pipelines
SVM (TF-IDF + class-balanced SVC)
•	Text lemmatization & stop-word removal
•	TF-IDF (1–2 n-grams, configurable features)
•	SVC with class_weight='balanced'
•	GridSearchCV across C, gamma, TF-IDF features
•	Metrics: accuracy, weighted precision/recall
•	Artifacts: learning curves, confusion matrix (saved to results/)
Sentence embeddings (USE)
•	Pretrained Universal Sentence Encoder (TF Hub) as frozen encoder
•	Dense head + dropout
•	EarlyStopping & class weights for imbalance
•	Metrics & confusion matrices exported to results/

## Results

| Model        | Accuracy | Precision (w) | Recall (w) |
| ------------ | -------- | ------------- | ---------- |
| TF-IDF + SVM | 0.XX     | 0.XX          | 0.XX       |
| USE + Dense  | 0.XX     | 0.XX          | 0.XX       |

Generated figures:
•	results/learning_curves/svm_learning_curve.png
•	results/confusion_matrices/svm_confusion_matrix.png
•	results/confusion_matrices/use_confusion_matrix.png

Roadmap
•	Move current Colab code into src/ modules
•	Add CLI args & config (YAML) for experiments
•	Save trained models (artifacts/)
•	Add tests for preprocessing
•	Add Dockerfile + GitHub Actions (CI)
•	Extend to domain-specific transformers (BioBERT, PubMedBERT)

Cite / Thesis
If you use this repository in academic work, please cite the accompanying thesis:
Georgakopoulos I. (2024). Development of an Effective and Comprehensive Information Retrieval Method for the Impact of Biological Factors on Patients with Inflammatory Bowel Diseases using Data Mining Techniques. National and Kapodistrian University of Athens.

License
This project is licensed under the MIT License. See LICENSE.

Acknowledgments
•	PubMed 200k RCT dataset authors
•	Open-source libraries: scikit-learn, TensorFlow, TensorFlow Hub, NLTK, Matplotlib/Seaborn

