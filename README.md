# EU AI Act Compliance Classifier
### AI System Risk Assessment using Machine Learning

---

## What This Project Does

This project builds and evaluates a machine learning classifier that assesses AI systems against **EU AI Act compliance criteria** — classifying them as:

- 🟢 **Launch Ready** — meets key compliance requirements
- 🟡 **Needs Improvement** — gaps exist, fixable before launch
- 🔴 **Not Compliant** — serious violations, cannot launch

Built as an **AI testing and evaluation portfolio piece** — not a production compliance tool.

---

## Why I Built This

The EU AI Act compliance deadline is **August 2, 2026**.

Research shows:
- 78% of organisations have not taken meaningful steps toward compliance
- Only 5% of AI systems in our dataset were genuinely Launch Ready
- A full compliance assessment costs €3,000–€10,000 — unaffordable for most early-stage startups

This project explores whether machine learning can automate the **first-pass screening** — reducing the time and cost of initial compliance assessment.

---

## What's Included

```
📁 Repository Structure

ai_risk_classifier.py          → Full ML pipeline
                                 Data generation, training,
                                 tuning, evaluation, charts

eu_act_readiness_chatbot.html  → Interactive browser tool
                                 10-question EU AI Act assessment
                                 Prioritized action list per answer
                                 No data collected or stored

results/
  startup_compliance_results.png → 4 evaluation charts
  defect_report_finding_1.md     → Formal defect report
  defect_report_finding_3.md     → Formal defect report

README.md                      → This file
```

---

## The 10 EU AI Act Features

Each AI system is assessed against 10 criteria mapped directly to EU AI Act articles:

| Feature | EU AI Act Article |
|---|---|
| Used in critical sector | Article 6 + Annex III |
| Makes autonomous decisions | Article 14 |
| Has human oversight | Article 14 |
| Processes sensitive data | Article 10 |
| Training data documented | Article 10 |
| Has transparency policy | Article 13 |
| Decisions explainable | Article 13 |
| Risk assessment complete | Article 9 |
| Traces to AI Act articles | Article 9 |
| Has complaint process | Article 17 |

---

## Model Results — Honest Evaluation

| Metric | Result |
|---|---|
| Algorithm | Random Forest Classifier |
| Training set | 180 systems (60%) |
| Validation set | 60 systems (20%) |
| Test set | 60 systems (20%) |
| Best validation accuracy | 81.7% (10 trees) |
| **Final test accuracy** | **70.0%** |
| Gap (overfitting indicator) | 11.7% ⚠️ |

### Defects Found and Documented

**Finding 1 — High Severity**
> Model failed to correctly classify any startup as Launch Ready.
> F1 score: 0% on Launch Ready category.
> Root cause: Class imbalance — only 8 Launch Ready examples in 180 training samples.

**Finding 3 — High Severity**
> Overfitting detected — gap between validation and test accuracy exceeds 5% threshold.
> Validation: 81.7% → Test: 70.0% → Gap: 11.7%
> Root cause: Model tuned to validation patterns, struggled on truly unseen data.

---

## The Tester's Perspective

This project was approached as a ** exercise**, not a model-building exercise.

My contributions as a test lead:

- ✅ Defined the problem and acceptance criteria
- ✅ Identified and mapped 10 EU AI Act compliance features
- ✅ Designed the scoring and classification logic
- ✅ Challenged distribution assumptions with real research
- ✅ Spotted class imbalance risk before model training
- ✅ Wrote structured defect reports with root cause analysis
- ✅ Made the fitness-for-purpose decision (awareness tool, not legal advice)
- ✅ Designed the chatbot with deliberate safety boundaries (no verdict, no data collection, GDPR-safe)

---

## How to Run the Classifier

**Requirements:**
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Run:**
```bash
python ai_risk_classifier.py
```

Output: Console results + 4 charts saved as PNG

---

## How to Use the Chatbot

1. Download `eu_act_readiness_chatbot.html`
2. Open in any browser
3. Answer 10 yes/no questions about your AI system
4. Receive prioritized action list based on EU AI Act articles

**No installation needed. No data collected. Works offline.**

---

## Important Disclaimer

> This tool was built to help AI startups in Europe understand where they stand before the August 2, 2026 EU AI Act deadline — without needing to spend €3,000–€10,000 on a consultant for an initial assessment.
It is based on synthetic data and achieves 70% accuracy — documented honestly in the defect reports above. It is designed as a first-pass awareness tool, not a replacement for legal advice. It is trained on synthetic (fictional) data and achieves 70% test accuracy with documented limitations.
>
> **It does not constitute legal advice and should never replace a qualified EU AI Act compliance assessment.** For binding compliance decisions, consult a qualified EU AI Act specialist.

---

## Built With

- **Python** — scikit-learn, pandas, numpy, matplotlib, seaborn
- **HTML / JavaScript** — interactive chatbot (no frameworks)
- **Claude AI (Anthropic)** — used as an AI-assisted development tool for implementation. Problem definition, EU AI Act research, testing methodology, defect reports, and all design decisions were made by the author.

---

## About the Author

Software Test Lead with 5 years of enterprise testing experience, specialising in AI system evaluation and EU AI Act compliance testing. 

**Open to freelance AI testing and compliance review work in Germany.**

📧 Connect on LinkedIn: https://www.linkedin.com/in/hira-khalidd/

---

## License

MIT License — free to use, adapt, and share with attribution.
