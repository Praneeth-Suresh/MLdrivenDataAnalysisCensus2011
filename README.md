# Exploring India's 2011 Census employment tables

This learning project reshapes the Census 2011 B-1 population counts by age and sex, fits a small TensorFlow model, and inspects its input gradients. Start with [model.py](model.py) and the checked-in [B-1 workbook](DDWCT-0000B-01.xls).

## What the analysis can show

Pandas groups the workbook into a model input table. TensorFlow gradients then show how the fitted model's output responds to small changes in the inputs. These are model sensitivities, not causal effects. The script uses workbook counts; it does not calculate or validate an unemployment rate. Treat this as an exploratory snapshot, not a statistical or policy conclusion.

## Running the historical script

The script reads the workbook next to itself, regardless of the current working directory. Its imports require pandas, xlrd (for the legacy .xls workbook), NumPy, Matplotlib, TensorFlow and Keras. The original dependency versions were not recorded. Training may require a compatible legacy TensorFlow/Keras environment; a current clean-environment end-to-end run has not been verified.
