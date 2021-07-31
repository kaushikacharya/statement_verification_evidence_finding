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
         python -u -m src.dataset --data_dir <data_dir> --data_split <data_split> --submit_dir <submit_dir> --flag_cell_span --flag_cell_span --flag_approx_string_match
    ```
    
- Evaluate metrics
    ```python
         python -u -m src.evaluate --input_dir <input_dir> --output_dir ./output/score/ --statistics_dir ./output/statistics --data_split <data_split>
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
| task_a_2way_f1 |  0.3493 | 0.3564 | 0.3770 |
| task_a_3way_f1 |  0.3493 | 0.4174 | 0.4695 |
| task_b_f1      |   NA    | 0.3538 | 0.3641 |

#### Publication
[KaushikAcharya at SemEval-2021 Task 9: Candidate Generation for Fact Verification over Tables](https://aclanthology.org/2021.semeval-1.181/) (SemEval)

#### Citation
If you find this implementation helpful, please consider citing:
```
@inproceedings{acharya-2021-kaushikacharya,
    title = "{K}aushik{A}charya at {S}em{E}val-2021 Task 9: Candidate Generation for Fact Verification over Tables",
    author = "Acharya, Kaushik",
    booktitle = "Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.semeval-1.181",
    doi = "10.18653/v1/2021.semeval-1.181",
    pages = "1271--1275",
    abstract = "This paper describes the system submitted in the SemEval-2021 Statement Verification and Evidence Finding with Tables task. The system relies on candidate generation for logical forms on the table based on keyword matching and dependency parsing on the claim statements.",
}
```
