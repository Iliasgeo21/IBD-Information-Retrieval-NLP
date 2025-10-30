from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ——— keywords for IBD & anti-TNF filtering ———
IBD_TERMS = [
    'inflammatory bowel disease', 'IBD', "Crohn's disease", 'Ulcerative colitis',
    'colitis', 'ileitis', 'IBS', 'enteritis', 'intestinal microbiota',
    'microbiome', 'gastrointestinal tract', 'mucosal inflammation',
    'mucosal', 'gut microbiota', 'gut', 'Th1', 'Th2', 'Th17',
    'Regulatory T cells', 'colon', 'small intestine', 'stool', 'rectum', 'pancolitis'
]
ANTI_TNF_TERMS = [
    'anti-TNF', 'infliximab', 'adalimumab', 'certolizumab', 'etanercept',
    'golimumab', 'TNF inhibitors', 'Tumor Necrosis Factor',
    'anti-tumor necrosis factor', 'TNF blocker', 'humira', 'simponi',
    'cimzia', 'remicade', 'anti-TNF monoclonal antibodies', 'TNF receptor',
    'anti-cytokine therapy', 'TNF-alpha', 'immunotherapy', 'biosimilar',
    'biologic drug', 'autoimmune disease treatment', 'anti-inflammatory biologics'
]

def _ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

def get_lines(filename: str | Path) -> list[str]:
    with open(filename, 'r', encoding='utf-8') as f:
        return f.readlines()

def parse_pubmed200k_file(filename: str | Path) -> list[dict]:
    """
    Διαβάζει το *.txt του PubMed 200k RCT και επιστρέφει list από dicts
    με πεδία: abstract_id, target, text, line_number, total_lines
    """
    input_lines = get_lines(filename)
    abstract_lines = ""
    samples = []
    abstract_id = "Undefined"

    for line in input_lines:
        if line.startswith("###"):
            abstract_id = line.strip()[3:]
            abstract_lines = ""
        elif line.isspace():
            if abstract_lines:
                split_lines = abstract_lines.splitlines()
                for i, raw in enumerate(split_lines):
                    parts = raw.split("\t")
                    if len(parts) < 2:
                        continue
                    samples.append({
                        "abstract_id": abstract_id,
                        "target": parts[0],
                        "text": parts[1].lower(),
                        "line_number": i,
                        "total_lines": len(split_lines) - 1
                    })
        else:
            abstract_lines += line
    return samples

def load_pubmed200k_rct(data_dir: str | Path):
    data_dir = Path(data_dir)
    train = pd.DataFrame(parse_pubmed200k_file(data_dir / "train.txt"))
    dev   = pd.DataFrame(parse_pubmed200k_file(data_dir / "dev.txt"))
    test  = pd.DataFrame(parse_pubmed200k_file(data_dir / "test.txt"))
    full_train = pd.concat([train, dev], ignore_index=True)
    all_data   = pd.concat([train, dev, test], ignore_index=True)
    return train, dev, test, full_train, all_data

def filter_by_keywords(df: pd.DataFrame,
                       ibd_terms: list[str] = IBD_TERMS,
                       anti_tnf_terms: list[str] = ANTI_TNF_TERMS) -> pd.DataFrame:
    keywords = ibd_terms + anti_tnf_terms
    
    escaped = [re.escape(k) for k in keywords]
    pattern = r'\b(?:' + '|'.join(escaped) + r')\b'
    return df[df["text"].str.contains(pattern, flags=re.IGNORECASE, na=False)]

def clean_texts(texts: list[str]) -> list[str]:
    _ensure_nltk()
    sw = set(stopwords.words('english'))
    lem = WordNetLemmatizer()
    out = []
    for doc in texts:
        words = []
        for w in doc.split():
            wl = w.lower()
            if wl in sw:
                continue
            words.append(lem.lemmatize(wl))
        out.append(" ".join(words))
    return out
