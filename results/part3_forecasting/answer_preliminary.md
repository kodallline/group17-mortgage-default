**Ranking drifts over time.** The base model (XGBoost) had ROC AUC 0.791 on validation (2015), but 0.702 on the test (2017–2018) and 0.724 without the hurricane quarter.

**The level of risk drifts too.** The default rate was 0.071% in C1, 0.096% in C2 and 0.120% in the test (0.091% without the hurricane quarter). O/E on the test is 2.10 for the base model, 2.06 for the global calibration, 1.90 for the regional calibration. All three versions under-predict the future level of default.

**Global calibration** (platt, slope b = 0.93) changes the test log loss from 0.009233 to 0.009203. It is an increasing transformation, so the ranking and ROC AUC stay the same as for the base model.

**Regional calibration** (λ = 1, shifts for 11 states) has test log loss 0.009119 against 0.009203 for the global calibration (difference -0.000084). Without the 2017-09-30 quarter the difference is -0.000023. The Brier score barely moves (-1.3e-07 and -1.8e-08): with defaults this rare it mostly reflects the base rate. Florida and Texas alone give 97% of the total log-loss gain, and the hurricane quarter gives 77% of it.

**Answer.** On this future test, regional calibration with shrinkage gives slightly more accurate probabilities than one global calibration, and the sign holds without the hurricane quarter. The gain is small and shrinks about 4 times without that quarter. Participant 4's paired bootstrap over loans will show whether it is larger than test noise.
