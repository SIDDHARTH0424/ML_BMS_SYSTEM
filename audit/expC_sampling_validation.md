# Experiment C Mixed Sampler Validation (Part 5)

**Total Sampled Episodes**: 2,000  
**Sampling Boundary**: $T < 35.0^\circ\text{C}$ (Normal) vs $T \ge 35.0^\circ\text{C}$ (Stress)  
**Sampler Status**: VALIDATED  

---

## 1. Distribution Mixture Proportions

| Distribution | Target Probability | Observed Count | Observed Probability | Absolute Difference |
|---|---|---|---|---|
| **Normal** (15.0–35.0°C) | 0.75 (75.0%) | 1502 | 0.7510 (75.10%) | 0.0010 |
| **Stress** (35.0–45.0°C) | 0.25 (25.0%) | 498 | 0.2490 (24.90%) | 0.0010 |

---

## 2. Temperature Distribution Statistics

- **Minimum Ambient**: 15.01°C (Expected $\ge 15.0^\circ\text{C}$)
- **Maximum Ambient**: 44.98°C (Expected $\le 45.0^\circ\text{C}$)
- **Empirical Mean Ambient**: 28.85°C
- **Theoretical Expected Mean**: 28.75°C ($0.75 \times 25.0 + 0.25 \times 40.0$)

---

## 3. Scientific Finding

The mixed distribution sampler correctly realizes the specified 75%/25% two-component mixture over $N = 2,000$ independent draws. Observed stress fraction is 24.90% (difference of 0.10% vs target 25.00%), which is within binomial sampling error bounds.