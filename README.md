# Music Album Deduplication Using String Similarity

## Overview

This project tackles the problem of **semantic duplication** in large music metadata collections. Albums may appear multiple times with inconsistent metadata (e.g., different spellings, casing, or track listings), which negatively affects catalog quality and user experience. The goal is to **automatically detect duplicate albums** using string similarity metrics and evaluate the effectiveness against a ground truth dataset.

## Business & Research Questions

1. How can we automatically detect and flag duplicate music album entries with inconsistent metadata?  
2. How well does our model perform compared to the provided ground truth duplicates?

## Dataset

- `cddb_discs.xml`: Primary dataset of music albums, including artist, title, and tracklist.
- `cddb_9763_dups.xml`: Ground truth file listing known duplicate album pairs.

## Methodology

1. **Data Parsing**: XML files parsed using `xml.dom.minidom`, with fallback logic for missing or alternate fields (e.g., `id` vs `cid`).
2. **Preprocessing**: Lowercasing, whitespace normalization, and punctuation removal applied to all text fields.
3. **Similarity Metrics**:
   - Jaro-Winkler: Used for artist and title fields (via `jellyfish`)
   - Jaccard Similarity: Used for tracklists, treating them as unordered sets
4. **Composite Score**: Weighted average of similarity scores:
   - 40% artist similarity
   - 40% album title similarity
   - 20% tracklist similarity
5. **Thresholding**: Pairs with similarity ≥ 0.8 are labeled as duplicates.
6. **Evaluation**: Compared predicted pairs to ground truth using precision, recall, F1 score, and accuracy.

## Results

| Metric               | Value |
|----------------------|-------|
| Precision            | 0.99  |
| Recall               | 0.93  |
| F1 Score             | 0.96  |
| Duplicates Predicted | 280   |
| Ground Truth Pairs   | 298   |

- High precision ensures minimal false positives.
- Slightly lower recall reflects the model’s cautious approach to labeling uncertain matches.

## Technologies Used

- Python
- Jellyfish (string similarity)
- Scikit-learn (evaluation metrics)
- xml.dom.minidom (XML parsing)

## Key Script

- `dedublication.py`: Main pipeline that parses data, computes similarity scores, matches albums, and evaluates results.
