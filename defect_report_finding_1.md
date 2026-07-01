# Defect Report — Finding 1

**Project:** EU AI Act Compliance Classifier
**Date:** July 2026
**Reported by:** Software Test Lead

---

## Summary

Model failed to correctly classify any AI startup as Launch Ready in the test set.

---

## Details

| Field | Value |
|---|---|
| **Title** | Model failed to correctly classify any startup as Launch Ready |
| **Severity** | High |
| **Status** | Open |
| **Priority** | High — resolve before any public release |

---

## Description

The model scored 0% Precision, 0% Recall, and 0% F1 on the Launch Ready category — meaning it did not correctly identify a single Launch Ready startup across the entire test set of 60 systems.

This is a complete failure on one classification category, not a minor underperformance.

---

## Test Evidence

| Metric | Launch Ready | Needs Improvement | Not Compliant |
|---|---|---|---|
| Precision | **0%** | 57% | 78% |
| Recall | **0%** | 62% | 81% |
| F1 Score | **0%** | 59% | 79% |
| Test count | 3 | 21 | 36 |

All 3 Launch Ready startups in the test set were misclassified.

---

## Expected Result

Launch Ready F1 score should be at least **50%** — comparable to the model's weakest other category (Needs Improvement: 59%).

## Actual Result

Launch Ready F1 score: **0%**
All Launch Ready startups misclassified as Needs Improvement.

---

## Root Cause

Class imbalance in training data:

```
Training set distribution:
Not Compliant      : 110 systems (61%) ← learned well
Needs Improvement  :  62 systems (34%) ← learned reasonably
Launch Ready       :   8 systems  (4%) ← barely learned
```

With only 8 examples of Launch Ready in 180 training samples, the model had insufficient data to learn the patterns of this category reliably.

---

## Impact

- Model cannot identify genuinely compliant startups
- A Launch Ready startup may be incorrectly told it Needs Improvement
- Costs them unnecessary consultant time and fees
- Reduces trust in the tool's positive classifications

---

## Recommendations

1. Collect significantly more Launch Ready training examples — target minimum 50
2. Apply SMOTE (Synthetic Minority Oversampling Technique) to balance classes
3. Consider merging Launch Ready and Needs Improvement if more data cannot be obtained
4. Re-test after improvement — target F1 above 50% before next release

---

## Linked Findings

- Finding 3 — Overfitting gap (related root cause: insufficient data variety)
