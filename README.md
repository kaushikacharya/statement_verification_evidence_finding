# Statement Verification and Evidence Finding with Tables
This repository contains the solution code for the [shared task](https://sites.google.com/view/sem-tab-facts)
which is part of the [SemEval 2021 shared tasks](https://semeval.github.io/SemEval2021/tasks).

#### Tasks
- Task A: Statement Verification
- Task B: Evidence Finding

Task A evaluation:
- 2 way: Statements are considered only if its ground truth verification is either entailed or refuted.
- 3 way: Statements with unknown ground truth verification also considered. 

#### How to run?
Below are the example commands.

- Run over the dataset
    ```python
         python -u -m src.dataset --data_dir <data_dir> --flag_cell_span --submit_dir <submit_dir>
    ```
    
- Evaluate metrics
    ```python
         python -u -m src.evaluate --input_dir <input_dir> --output_dir ./output/score/ --output_csv <csv_file>
    ```
    
    Directory structure expected by evaluate script
    ```
    input_dir
    └── ref (ground truthed xml files)
    └── res (predicted xml files)
    ```
    
#### Results
F1-score averaged over the tables.

|  Metrics/Split | Train   |   Dev  |  Test  |
|----------------|---------|--------|--------|
| # tables       |  981    |   52   |  52    |
| task_a_2way_f1 |  0.3584 | 0.3746 | 0.3921 |
| task_a_3way_f1 |  0.3584 | 0.4284 | 0.4832 |
| task_b_f1      |   NA    | 0.3645 | 0.3763 |

