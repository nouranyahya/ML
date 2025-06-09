# Model Comparison Report

## Executive Summary

**Recommended Model:** support_vector_machine_2025-06-09_15-12-28
**Overall Score:** 0.9148

## Performance Comparison

| Model                                      |   Accuracy |   Precision |   Recall |   F1-Score |   Prediction Time (s) |   ROC AUC |   Log Loss |
|:-------------------------------------------|-----------:|------------:|---------:|-----------:|----------------------:|----------:|-----------:|
| k-nearest_neighbors_2025-06-09_15-12-28    |      0.825 |      0.9114 |   0.7556 |     0.7892 |                0.0116 |    0.9814 |     0.3814 |
| support_vector_machine_2025-06-09_15-12-28 |      0.95  |      0.9692 |   0.9778 |     0.9716 |                0.0111 |    1      |     0.1697 |
| logistic_regression_2025-06-09_15-12-28    |      0.925 |      0.9623 |   0.8822 |     0.9129 |                0.0109 |    0.9987 |     0.3388 |
| random_forest_2025-06-09_15-12-29          |      0.975 |      0.9895 |   0.9818 |     0.9851 |                0.0413 |    1      |     0.1966 |

## Overall Ranking

1. **support_vector_machine_2025-06-09_15-12-28** (Score: 0.9148)
2. **logistic_regression_2025-06-09_15-12-28** (Score: 0.8825)
3. **k-nearest_neighbors_2025-06-09_15-12-28** (Score: 0.7893)
4. **random_forest_2025-06-09_15-12-29** (Score: 0.7840)

## Best Performers by Metric

- **Accuracy:** random_forest_2025-06-09_15-12-29 (0.9750)
- **Precision Macro:** random_forest_2025-06-09_15-12-29 (0.9895)
- **Recall Macro:** random_forest_2025-06-09_15-12-29 (0.9818)
- **F1 Macro:** random_forest_2025-06-09_15-12-29 (0.9851)
- **Roc Auc:** support_vector_machine_2025-06-09_15-12-28 (1.0000)
- **Prediction Time:** logistic_regression_2025-06-09_15-12-28 (0.0109)
- **Log Loss:** support_vector_machine_2025-06-09_15-12-28 (0.1697)

## Recommendations

### Primary Recommendation: support_vector_machine_2025-06-09_15-12-28


### Use Case Specific Recommendations

**Production Deployment:**
- Model: logistic_regression_2025-06-09_15-12-28
- Reason: Fastest prediction time for real-time applications

**Highest Accuracy:**
- Model: random_forest_2025-06-09_15-12-29
- Reason: Best overall accuracy for critical decisions

**Balanced Performance:**
- Model: random_forest_2025-06-09_15-12-29
- Reason: Best F1-score for balanced precision and recall

**Overall Best:**
- Model: support_vector_machine_2025-06-09_15-12-28
- Reason: Best combination of accuracy, F1-score, and speed

### Important Considerations

- Different models excel in different aspects - consider your priorities
- No statistically significant differences found - any model may be suitable
- Models with >90% accuracy: support_vector_machine_2025-06-09_15-12-28, logistic_regression_2025-06-09_15-12-28, random_forest_2025-06-09_15-12-29

## Statistical Significance

No statistically significant differences found between models.