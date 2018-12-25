

| **Algorithm** | **Best Parameters** | **Avg Precision** | **Avg Recall** | **Avg F1** | **Precision Score** |
| --- | --- | --- | --- | --- | --- |
| SVC | {&#39;C&#39;: 100, &#39;gamma&#39;: 0.001, &#39;kernel&#39;: &#39;rbf&#39;} | 0.96 | 0.96 | 0.96 | 0.9701492537313433 |
| DecisionTreeClassifier | {&#39;max\_depth&#39;: 100, &#39;max\_features&#39;: &#39;log2&#39;, &#39;min\_samples\_leaf&#39;: 5, &#39;min\_samples\_split&#39;: 10} | 0.94 | 0.94 | 0.94 | 0.9411764705882353 |
| MLPClassifier | {&#39;activation&#39;: &#39;tanh&#39;, &#39;alpha&#39;: 0.0001, &#39;hidden\_layer\_sizes&#39;: (10,), &#39;max\_iter&#39;: 200} | 0.95 | 0.95 | 0.95 | 0.9552238805970149 |
| GaussianNB | {} | 0.90 | 0.90 | 0.90 | 0.9242424242424242 |
| LogisticRegression | {&#39;fit\_intercept&#39;: True, &#39;max\_iter&#39;: 10, &#39;penalty&#39;: &#39;l1&#39;, &#39;tol&#39;: 0.0001} | 0.96 | 0.96 | 0.96 | 0.9558823529411765 |
| KNeighborsClassifier | {&#39;algorithm&#39;: &#39;ball\_tree&#39;, &#39;n\_neighbors&#39;: 10, &#39;p&#39;: 1, &#39;weights&#39;: &#39;uniform&#39;} | 0.96   | 0.96   | 0.96   | 0.9558823529411765 |
| BaggingClassifier | {&#39;max\_features&#39;: 0.5, &#39;max\_samples&#39;: 1.0, &#39;n\_estimators&#39;: 20, &#39;random\_state&#39;: None} | 0.97 | 0.97 | 0.97 | 0.9705882352941176 |
| RandomForestClassifier | {&#39;criterion&#39;: &#39;entropy&#39;, &#39;max\_depth&#39;: 200, &#39;max\_features&#39;: 0.5, &#39;n\_estimators&#39;: 20} | 0.96 | 0.96 | 0.96 | 0.9701492537313433 |
| AdaBoostClassifier | {&#39;algorithm&#39;: &#39;SAMME&#39;, &#39;learning\_rate&#39;: 0.8, &#39;n\_estimators&#39;: 200, &#39;random\_state&#39;: None} | 0.98 | 0.98 | 0.98 | 0.9850746268656716 |
| GradientBoostingClassifier | {&#39;loss&#39;: &#39;deviance&#39;, &#39;max\_depth&#39;: 3, &#39;max\_features&#39;: &#39;log2&#39;, &#39;n\_estimators&#39;: 200} | 0.97   | 0.97   | 0.97   | 0.9705882352941176 |
| XGBClassifier | {&#39;booster&#39;: &#39;gbtree&#39;, &#39;learning\_rate&#39;: 0.1, &#39;max\_delta\_step&#39;: 0, &#39;min\_child\_weight&#39;: 1} | 0.97   | 0.97   | 0.97   | 0.9705882352941176 |

The best algorithm among these was the AdaBoost classifier.
