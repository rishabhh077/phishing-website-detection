Data files are git-ignored (too large to commit) and regenerated with:

    git clone --depth 1 https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs.git data/ml_urls
    python src/build_dataset.py   # -> data/raw_urls.csv
    python src/train.py           # -> data/processed_features.csv (cached) + models/
