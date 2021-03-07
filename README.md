# Statement Verification and Evidence Finding with Tables
This repository contains the solution code for the [shared task](https://sites.google.com/view/sem-tab-facts)
which is part of the [SemEval 2021 shared tasks](https://semeval.github.io/SemEval2021/tasks).

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
