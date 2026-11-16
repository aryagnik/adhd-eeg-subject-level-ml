# Model comparison metrics

| Feature Set   | Model               |   Accuracy |   Balanced Accuracy |   Sensitivity |   Specificity |   ROC AUC |
|:--------------|:--------------------|-----------:|--------------------:|--------------:|--------------:|----------:|
| Spectral      | Dummy               |     0.5041 |              0.5000 |        1.0000 |        0.0000 |    0.3499 |
| Spectral      | Logistic Regression |     0.6033 |              0.6026 |        0.6885 |        0.5167 |    0.6443 |
| Spectral      | Random Forest       |     0.6446 |              0.6428 |        0.8689 |        0.4167 |    0.6948 |
| Spectral      | SVM                 |     0.5785 |              0.5762 |        0.8525 |        0.3000 |    0.6216 |
| Multi-domain  | Dummy               |     0.5041 |              0.5000 |        1.0000 |        0.0000 |    0.3499 |
| Multi-domain  | Logistic Regression |     0.7769 |              0.7764 |        0.8361 |        0.7167 |    0.8003 |
| Multi-domain  | Random Forest       |     0.7769 |              0.7766 |        0.8033 |        0.7500 |    0.8626 |
| Multi-domain  | SVM                 |     0.7769 |              0.7760 |        0.8852 |        0.6667 |    0.8372 |
