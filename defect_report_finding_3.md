# Defect Report — Finding 3

**Project:** EU AI Act Compliance Classifier
**Date:** July 2026
**Reported by:** Software Test Lead

---

## Summary

Overfitting detected — gap between validation and test accuracy exceeds the 5% acceptable threshold by 6.7%.

---

## Details

| Field | Value |
|---|---|
| **Title** | Overfitting detected — gap between validation and test accuracy exceeds acceptable threshold |
| **Severity** | High |
| **Status** | Open |
| **Priority** | High — resolve before public release |

---

## Description

The model shows a significant gap between validation accuracy (seen during tuning) and test accuracy (truly unseen data), indicating the model has overfit to the validation data and does not generalise well to real-world inputs.

---

## Steps to Reproduce

1. Train Random Forest model with 10 trees on training set (180 systems)
2. Evaluate on validation set — record accuracy
3. Evaluate on locked test set — record accuracy
4. Calculate gap between the two results

---

## Test Evidence

| Metric | Value |
|---|---|
| Validation accuracy | 81.7% |
| Test accuracy | 70.0% |
| Gap | **11.7%** |
| Acceptable threshold | 5% |
| Threshold breached by | **6.7%** |

---

## Expected Result

Gap between validation and test accuracy should be **less than 5%**, indicating a healthy, generalising model.

## Actual Result

Gap is **11.7%** — more than double the acceptable threshold.

---

## Testing Analogy

This is the ML equivalent of a system that works perfectly in the QA environment but behaves differently in UAT. The model tuned itself to patterns in the validation data. When it met truly unseen test data, performance dropped significantly — same as a system that passes regression testing but fails in production.

---

## Root Cause

The model was tuned against a single validation split of 60 systems. With this small sample and class imbalance (only 8 Launch Ready examples in training), the model learned some noise patterns specific to the validation set rather than truly generalising rules.

---

## Impact

- Real-world performance (70%) is significantly lower than validation suggested (81.7%)
- Users relying on validation accuracy would have an inflated expectation of reliability
- 3 out of 10 assessments will be incorrect on new, unseen AI systems
- Credibility of the tool is reduced if accuracy is overstated

---

## Recommendations

1. Apply cross-validation (5-fold or 10-fold) instead of single validation split — gives more reliable tuning results
2. Collect more training data — especially Launch Ready examples (currently only 8)
3. Try simpler models (Logistic Regression) which are less prone to overfitting on small datasets
4. Re-evaluate after data improvement — target test accuracy above 80% and gap below 5%

---

## Linked Findings

- Finding 1 — Launch Ready F1 score 0% (same root cause: insufficient data)
