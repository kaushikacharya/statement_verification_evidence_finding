#!/usr/bin/env python

"""
Pre-requisite:
    - src/dataset.py should be executed before src/evaluate.py
    - Ensure that there's no failed tables as mentioned in "summary" section of output of src/dataset.py
        - Otherwise evaluate script would fail.

Example command (to run on train data):
    python -u -m src.evaluate --data_split train > ./output/score/score_train.txt 2>&1
"""

import argparse
from csv import writer
from io import StringIO
import io
import pandas as pd
import sys
import os
import os.path
import numpy as np
from glob import glob
from bs4 import BeautifulSoup
import sklearn
import sklearn.metrics
import pdb

###USEFUL FUNCTION

def ls(filename):
    return sorted(glob(filename))

def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)

if os.name == "nt":
    filesep = '\\'
else:
    filesep = '/'

def read_xml(filename, type="prediction", verbose=False):
    ''' Read xml file and convert to solution dicts '''

    result_dict = {}
    content = open(filename, encoding="utf-8").read()
    soup = BeautifulSoup(content, "lxml")
    if verbose:
        print("Reading file {}".format(filename))

    # iterate over the tables
    for table in soup.find_all("table"):
        evidence_missing_flag = False
        evidence_repeat_flag = False
        if verbose:
            print("Reading table {} in file {}".format(table["id"], filename))
        result_dict[table["id"]] = {}
        # https://stackoverflow.com/questions/5015483/test-if-an-attribute-is-present-in-a-tag-in-beautifulsoup (Lucas S.'s answer)
        for statement in table.find_all("statement"):
            result_dict[table["id"]][statement["id"]] = {"type": statement["type"], "evidence":{}, "text": statement["text"] if statement.has_attr("text") else None}
        for row in table.find_all("row"):
            for cell in row.find_all("cell"):
                for evidence in cell.find_all("evidence"):
                    #if type == "prediction":
                    #    pdb.set_trace()
                    if type == "prediction" and (evidence["version"] == "" and evidence["type"] == ""):
                        if not evidence_missing_flag:
                            print("Evidence missing for table {} in file {}".format(table["id"], filename))
                            evidence_missing_flag = True
                        continue

                    if type=="prediction" and int(evidence["version"])>0:
                        if not evidence_repeat_flag:
                            print("Evidence version > 0 detected in prediction for table {} in file {}, "
                            "ignoring this evidence".format(table["id"], filename))
                            evidence_repeat_flag = True
                        continue

                    if evidence["version"] not in result_dict[table["id"]][evidence["statement_id"]]["evidence"]:
                        result_dict[table["id"]][evidence["statement_id"]]["evidence"][evidence["version"]] = {}


                    result_dict[table["id"]][evidence["statement_id"]]["evidence"][evidence["version"]][(cell["row-start"],
                                                                                    cell["col-start"])] = evidence["type"]

    return result_dict

def compare_evidences(gt, pred, statement_id, table_id):
    gt_res = []
    pred_res = []
    if len(gt) != len(pred):
        raise ValueError(
            "Evidence not provided for all cells in statement {} for table id = {} ".format(len(statement_id), len(table_id)))
    for coord, type in gt.items():
        gt_res.append(gt[coord]=="relevant")
        pred_res.append(pred[coord]=="relevant")

    return sklearn.metrics.f1_score(gt_res, pred_res)


def f1_scoring(res_dict):
    return sklearn.metrics.f1_score(res_dict["gt"], res_dict["pred"], average="micro")

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    scores_task_a_2way_list = []
    scores_task_a_3way_list = []
    scores_task_b_f1 = []

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = os.path.join(output_dir, 'scores.txt')
        output_file = open(output_filename, 'w')

        solution_names = sorted(ls(os.path.join(input_dir, 'ref', '*.xml')))
        prediction_files = ls(os.path.join(submit_dir, '*.xml'))

        if len(prediction_files) == 0:
            if len(glob(os.path.join(submit_dir, '*/*.xml'))) > 0:
                submit_dir_tmp = glob(os.path.join(submit_dir, '*/'))[0]
                if len(glob(os.path.join(submit_dir_tmp, '*/*.xml'))) == 0:
                    submit_dir = glob(os.path.join(submit_dir, '*/'))[1]
                else:
                    submit_dir = submit_dir_tmp
            else:
                raise IOError('No solution XML files detected in folder, please check your submission')

        csv_table_output = StringIO()
        csv_table_writer = writer(csv_table_output)
        csv_statement_output = StringIO()
        csv_statement_writer = writer(csv_statement_output)

        task_b_missing_flag = False
        # iterate over each of the documents
        for i, solution_file in enumerate(solution_names):
            scores_task_a_2way_list_file = []
            scores_task_a_3way_list_file = []
            scores_task_b_f1_file = []
            # Extract the dataset name from the file name
            basename = solution_file[-solution_file[::-1].index(filesep):-solution_file[::-1].index('.') - 1]

            try:
                predict_file = ls(os.path.join(submit_dir, basename + '*.xml'))[-1]
            except IndexError:
                raise IOError('Missing prediction file {}, Not found in submission directory {}'.format(basename, submit_dir))

            predict_name = predict_file[-predict_file[::-1].index(filesep):-predict_file[::-1].index('.') - 1]
            # Read the solution and prediction values into numpy arrays
            solution = read_xml(solution_file, type="solution", verbose=args.verbose)
            prediction = read_xml(predict_file, type="prediction", verbose=args.verbose)

            if len(solution) != len(prediction):
                raise ValueError(
                "There are {} tables for this file but only {} are predicted".format(len(solution), len(prediction)))

            # iterate over each of the table in the current document
            for table_id, table_res in solution.items():
                if table_id not in prediction:
                    raise ValueError(
                        "Table with id {} is missing".format(table_id))
                if len(prediction[table_id]) != len(table_res):
                    raise ValueError(
                        "There are {} statements for table id = {} but only {} are predicted".format(len(table_res), table_id,
                                                                                             len(prediction[table_id])))
                scores_task_a_2way_table_list = {"gt":[], "pred": []}
                scores_task_a_3way_table_list = {"gt":[], "pred": []}
                scores_task_b_f1_table = []

                # iterate over each of the statement in the current table
                for statement_id, statement_res in table_res.items():
                    if statement_id not in prediction[table_id]:
                        raise ValueError(
                            "Statement with id {} in Table with id {} is missing".format(statement_id, table_id))

                    # Append results
                    if statement_res["type"] == "entailed":
                        scores_task_a_2way_table_list["gt"].append(1)
                        scores_task_a_3way_table_list["gt"].append(1)

                    elif statement_res["type"] == "refuted":
                        scores_task_a_2way_table_list["gt"].append(0)
                        scores_task_a_3way_table_list["gt"].append(0)
                    else:
                        scores_task_a_3way_table_list["gt"].append(2)

                    # Append pred results
                    if statement_res["type"] == "entailed" or statement_res["type"] == "refuted":
                        if prediction[table_id][statement_id]["type"]=="entailed":
                            scores_task_a_2way_table_list["pred"].append(1)
                            scores_task_a_3way_table_list["pred"].append(1)
                        elif prediction[table_id][statement_id]["type"]=="refuted":
                            scores_task_a_2way_table_list["pred"].append(0)
                            scores_task_a_3way_table_list["pred"].append(0)
                        elif prediction[table_id][statement_id]["type"]=="unknown":
                            scores_task_a_2way_table_list["pred"].append(2)
                            scores_task_a_3way_table_list["pred"].append(2)
                        else:
                            raise ValueError(
                                "Only entailed/refuted/unknown relationships allowed, {} "
                                "found instead".format(prediction[table_id][statement_id]["type"]))
                    else:
                        if prediction[table_id][statement_id]["type"]=="entailed":
                            scores_task_a_3way_table_list["pred"].append(1)
                        elif prediction[table_id][statement_id]["type"]=="refuted":
                            scores_task_a_3way_table_list["pred"].append(0)
                        elif prediction[table_id][statement_id]["type"] == "unknown":
                            scores_task_a_3way_table_list["pred"].append(2)
                        else:
                            raise ValueError(
                                "Only entailed/refuted/unknown relationships allowed, {} "
                                "found instead".format(prediction[table_id][statement_id]["type"]))

                    if len(prediction[table_id][statement_id]["evidence"]) == 0 and len(solution[table_id][statement_id]["evidence"]) > 0:
                        task_b_missing_flag = True
                    elif len(prediction[table_id][statement_id]["evidence"]) > 0:
                        f1_scores = []
                        for version, evidence_res in statement_res["evidence"].items():
                            f1_scores.append(compare_evidences(evidence_res,
                                                               prediction[table_id][statement_id]["evidence"]["0"],
                                                               statement_id, table_id))

                        scores_task_b_f1_table.append(max(f1_scores))

                    if statement_res["type"] != prediction[table_id][statement_id]["type"]:
                        csv_statement_row_data = [os.path.basename(solution_file), table_id, statement_id, statement_res["type"],
                                                  prediction[table_id][statement_id]["type"], statement_res["text"]]
                        csv_statement_writer.writerow(csv_statement_row_data)

                # compute F1 score for the current table
                twoway_f1 = f1_scoring(scores_task_a_2way_table_list)
                threeway_f1 = f1_scoring(scores_task_a_3way_table_list)
                print("Results for file: {} table: {}".format(solution_file, table_id))
                print("task_a_2way_f1_total: %f " % (twoway_f1))
                print("task_a_3way_f1_total: %f " % (threeway_f1))
                scores_task_a_2way_list_file.append(twoway_f1)
                scores_task_a_3way_list_file.append(threeway_f1)

                csv_row_data = [os.path.basename(solution_file), table_id, "{:.3f}".format(twoway_f1), "{:.3f}".format(threeway_f1), len(scores_task_a_3way_table_list["gt"])]
                csv_table_writer.writerow(csv_row_data)

                if task_b_missing_flag and len(scores_task_b_f1_table) > 0:
                    raise ValueError("Some evidence is missing for table {}, but some are also provided. "
                                     "If participating in evidence task, all evidence must be provided, "
                                     "else there must be no evidence.".format(table_id))
                if len(scores_task_b_f1_table) > 0:
                    scores_task_b_f1_file.append(np.mean(scores_task_b_f1_table))

            # extend dataset scores with the document scores
            scores_task_a_2way_list += scores_task_a_2way_list_file
            scores_task_a_3way_list += scores_task_a_3way_list_file
            scores_task_b_f1 += scores_task_b_f1_file

            if len(scores_task_b_f1_file) > 0:
                print("task_b_f1_total: %f" % (np.mean(scores_task_b_f1_file)))


        print("\n\nOverall Results")
        output_file.writelines(["task_a_2way_f1_total: %.4f \n" % (np.mean(scores_task_a_2way_list))])
        print("task_a_2way_f1_total: %.4f \n" % (np.mean(scores_task_a_2way_list)))
        output_file.writelines(["task_a_3way_f1_total: %.4f \n" % (np.mean(scores_task_a_3way_list))])
        print("task_a_3way_f1_total: %.4f \n" % (np.mean(scores_task_a_3way_list)))
        if len(scores_task_b_f1) == 0:
            scores_task_b_f1 = [0]
        output_file.writelines(["task_b_f1_total: %.4f \n" % (np.mean(scores_task_b_f1))])
        print("task_b_f1_total: %.4f \n" % (np.mean(scores_task_b_f1)))
        output_file.close()

        # Write table results in a csv
        csv_table_output.seek(0)
        df_columns = ["File", "Table", "Two-way F1", "Three-way F1", "n_statements"]
        results_df = pd.read_csv(filepath_or_buffer=csv_table_output, names=df_columns)

        # https://stackoverflow.com/questions/3191528/csv-in-python-adding-an-extra-carriage-return-on-windows
        #   - Explains why newline="" is needed to avoid additional newline character
        output_table_csv = "score_table_{}_df.csv".format(args.data_split)
        with io.open(os.path.join(output_dir, output_table_csv), mode="w", newline="") as fd:
            results_df.to_csv(path_or_buf=fd, index=False)

        # Write statement level results in a csv
        csv_statement_output.seek(0)
        df_columns = ["File", "Table", "StatementId", "ground_truth_type", "predict_type", "text"]
        results_statement_df = pd.read_csv(filepath_or_buffer=csv_statement_output, names=df_columns)

        if not os.path.exists(args.statistics_dir):
            os.makedirs(args.statistics_dir)
        output_statement_csv = "type_statement_{}_df.csv".format(args.data_split)
        with io.open(os.path.join(args.statistics_dir, output_statement_csv), mode="w", newline="") as fd:
            results_statement_df.to_csv(path_or_buf=fd, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", action="store", default="C:/KA/data/NLP/statement_verification_evidence_finding/train_manual_v1.3.2/v1.3.2/",
                        dest="input_dir", help="Directory containing a) ground truth data in res subdirectory   b) predicted data in ref subdirectory")
    parser.add_argument("--output_dir", action="store", default=os.path.join(os.path.dirname(__file__), "../output/score"), dest="output_dir")
    parser.add_argument("--statistics_dir", action="store", default=os.path.join(os.path.dirname(__file__), "../output/statistics"), dest="statistics_dir")
    parser.add_argument("--data_split", action="store", default="train", dest="data_split",
                        help="data split to be executed. Values permitted: train, dev, test")
    parser.add_argument("--verbose", action="store_true", default=False, dest="verbose")
    args = parser.parse_args()

    print("args: {}".format(args))
    main(args=args)
