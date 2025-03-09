# MLdrivenDataAnalysisCensus2011
The following dataset from the Indian Census is analysed: "B-1 Main workers, Marginal workers, Non-workers and those marginal workers, non-workers seeking/available for work classified by age and sex". 

I used this data to determine the factor that has the greatest impact on unemployment rate.

## What problem does it solve?
It reorganises data so that useful data about the trends highlighted can be obtained, from a very descriptive dataset. More broadly, this algorithm solves the issue of information organisation. 
The goal of the algorithm is to propose a solution to get insights more readily available.

## How does it work?
The main tools used to achieve the data analysis are:
•	`pivot_table` in pandas – used to set columns more distinctly to help visualise trends more clearly
•	`GradientTape` in TensorFlow to figure out which field has the most impact on the outcome

## Inspiration
I was motivated to work on this project due to the great amount of power that understanding data promises, especially in a large country like India. Many social problems and inequities are reflected in data but we fail to see insights just from graphs drawn straight out of the data. This was my attempt to reorganise the information in dense datasets like the census so that decision making on issues can be better informed in the future.
