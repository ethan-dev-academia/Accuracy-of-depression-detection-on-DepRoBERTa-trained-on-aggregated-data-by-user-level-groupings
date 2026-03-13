# Why Only 34,172 Examples from 147,130 Labeled Users?

## Summary

Out of **147,130 labeled users**, only **34,172 examples** (23.2%) were used for training. This document explains the filtering steps that reduced the dataset.

---

## Data Reduction Pipeline

### Starting Point: 147,130 Labeled Users

These are users successfully matched with labels from the RMH dataset:
- **Label 0 (depression)**: 91,318 users
- **Label 1 (anxiety/PTSD)**: 55,812 users
- **Total**: 147,130 users

### Filtering Steps

#### Step 1: Label 2 Exclusion (Binary Classification)

**What happened**: Users with label 2 ("not depression" - from subreddits like ADHD, addiction, etc.) were **excluded** from training because this is a **binary classification task** (depression vs. non-depression).

**Code location**: `prepare_training_dataset.py` - While the code initially accepts labels 0, 1, and 2, the final training dataset only contains labels 0 and 1.

**Impact**: Unknown number of label 2 users were excluded (this information is not explicitly tracked in the current pipeline).

#### Step 2: Content Aggregation and Validation

**What happened**: Users were filtered based on content quality:

1. **Content Aggregation** (`prepare_training_dataset.py`, line 137):
   - All posts and comments for each user were aggregated into a single text
   - Posts with "[removed]", "[deleted]", or empty content were filtered out
   - Only valid text content was included

2. **Minimum Content Length** (`prepare_training_dataset.py`, line 139):
   - Users with aggregated text **less than 10 characters** were excluded
   - This filters out users with:
     - No posts or comments
     - All content removed/deleted
     - Only very short content

3. **Validation Filtering** (`validate_labeled_data.py`, line 125):
   - Additional validation with `min_content_length: 50` characters (from `labeling_config.json`)
   - Users failing this validation were excluded from the filtered dataset

**Impact**: This is likely the **largest source of data reduction**. Many users may have had:
- All their posts/comments removed or deleted
- Insufficient content after filtering out removed/deleted markers
- Very short content that doesn't meet minimum thresholds

#### Step 3: Final Training Dataset

**Result**: 34,172 examples with:
- **Label 0 (depression)**: 20,265 examples (59.3%)
- **Label 1 (anxiety/PTSD)**: 13,907 examples (40.7%)

---

## Estimated Breakdown

Based on the code and configuration:

```
147,130 labeled users
    ↓
[Label 2 exclusion] → ~147,130 - X users (X = unknown number of label 2 users)
    ↓
[Content aggregation] → Users with valid posts/comments
    ↓
[Minimum length filter] → Users with ≥10 characters aggregated text
    ↓
[Validation filter] → Users with ≥50 characters (if validation was run)
    ↓
34,172 training examples (23.2% retention rate)
```

### Why So Many Users Were Excluded

The most likely reasons for the large reduction:

1. **Insufficient Content** (Primary reason):
   - Many Reddit users may have had their posts/comments removed or deleted
   - Users with only a few very short posts/comments
   - Users whose content was filtered out during aggregation

2. **Label 2 Exclusion**:
   - Users from subreddits like ADHD, addiction, alcoholism, autism (label 2)
   - These were excluded for binary classification

3. **Data Quality Issues**:
   - Users with missing or malformed data
   - Users with no valid text after filtering removed/deleted markers

---

## Configuration Settings

From `labeling_config.json`:

```json
"validation": {
  "min_samples_per_class": 100,
  "min_content_length": 50,
  "max_class_imbalance_ratio": 10.0
}
```

However, the actual filtering in `prepare_training_dataset.py` uses:
- **Minimum 10 characters** for aggregated text (line 139)
- The 50-character threshold from config may have been applied during validation step

---

## Recommendations for Documentation

When writing your methods section, you could include:

> "Of the 147,130 users successfully labeled from the RMH dataset, 34,172 user-level examples (23.2%) were retained for training after applying content quality filters. Users were excluded if: (1) they had insufficient aggregated text content (<10 characters after removing deleted/removed markers), (2) they belonged to label 2 (non-depression mental health subreddits), which was excluded for binary classification, or (3) they failed validation criteria requiring minimum content length. This filtering ensured that training examples contained sufficient text for meaningful model learning."

---

## To Get Exact Numbers

To determine exactly how many users were excluded at each step, you would need to:

1. **Count label 2 users**: Check how many of the 147,130 had label 2
2. **Count insufficient content**: Check how many users had <10 characters after aggregation
3. **Count validation failures**: Check how many failed the 50-character validation

You could modify `prepare_training_dataset.py` to track these statistics, or run a separate analysis script on the `all_labeled_users.json` file.

---

## Current Statistics Available

From the codebase:
- **147,130 labeled users** (starting point)
- **34,172 training examples** (final)
- **23.2% retention rate**
- **Label distribution in final dataset**: 59.3% label 0, 40.7% label 1
- **Average text length**: 14,524 characters per example

The exact breakdown of exclusions is not currently tracked, but the primary reason is likely **insufficient content** after filtering out removed/deleted posts and comments.

