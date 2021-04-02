#!/usr/bin/env python

"""
N.B. If executing in console, execute the following as pre-requisite:
export PYTHONIOENCODING=UTF-8

Command: python -u -m src.dataset --flag_cell_span --flag_approx_string_match > ./output/dataset/dataset_train.txt 2>&1
"""

import argparse
import glob
import io
import os
import pandas as pd
import shutil
import traceback

from tqdm import tqdm

from src.document import Document
from src.nlp_process import NLPProcess

class Dataset:
    def __init__(self, nlp_process_obj):
        self.nlp_process_obj = nlp_process_obj

    def process_data_dir(self, data_dir, data_split, flag_cell_span=True, flag_approx_string_match=True, submit_dir=None):
        failed_tables_dataset = []
        n_tables_dataset = 0
        n_statements_dataset = 0
        n_statements_with_column_matched_dataset = 0
        n_statements_table_frequency_map_dataset = dict()
        table_statistics_dataset = {"doc": [], "table_id": [], "n_column_rows": [], "n_data_rows": []}
        confusion_dict_dataset = dict()
        n_files = 0

        if submit_dir:
            # First delete the submit directory if exists
            if os.path.exists(submit_dir):
                shutil.rmtree(submit_dir)
            # Then create the submit directory
            os.makedirs(submit_dir)

        for xml_file_path in glob.iglob(os.path.join(data_dir, "*.xml")): # tqdm(glob.glob())
            print("\n{}".format(xml_file_path))
            doc_obj = Document(nlp_process_obj=self.nlp_process_obj)
            doc_output_dict = doc_obj.parse_xml(xml_file=xml_file_path, flag_cell_span=flag_cell_span, flag_approx_string_match=flag_approx_string_match, verbose=False)

            failed_tables_doc = doc_output_dict["failed_tables_doc"]
            n_tables_doc = doc_output_dict["n_tables_doc"]
            n_statements_doc = doc_output_dict["n_statements_doc"]
            n_statements_table_frequency_map_doc = doc_output_dict["n_statements_table_frequency_map_doc"]
            n_statements_with_column_matched_doc = doc_output_dict["n_statements_with_column_matched_doc"]
            table_statistics_doc = doc_output_dict["table_statistics_doc"]
            doc_output_tree = doc_output_dict["doc_output_tree"]
            confusion_dict_doc = doc_output_dict["confusion_dict_doc"]

            if submit_dir:
                submit_filepath = os.path.join(submit_dir, os.path.basename(xml_file_path))
                with io.open(file=submit_filepath, mode="wb") as fd:
                    doc_output_tree.write(fd)

            if len(failed_tables_doc) > 0:
                failed_tables_dataset.append({os.path.basename(xml_file_path): failed_tables_doc})

            n_tables_dataset += n_tables_doc
            n_statements_dataset += n_statements_doc

            for n_statements in n_statements_table_frequency_map_doc:
                if n_statements not in n_statements_table_frequency_map_dataset:
                    n_statements_table_frequency_map_dataset[n_statements] = 0

                n_statements_table_frequency_map_dataset[n_statements] += n_statements_table_frequency_map_doc[n_statements]

            n_statements_with_column_matched_dataset += n_statements_with_column_matched_doc

            for table_id in table_statistics_doc:
                table_statistics_dataset["doc"].append(os.path.basename(xml_file_path))
                table_statistics_dataset["table_id"].append(table_id)
                table_statistics_dataset["n_column_rows"].append(table_statistics_doc[table_id]["n_column_rows"])
                table_statistics_dataset["n_data_rows"].append(table_statistics_doc[table_id]["n_data_rows"])

            for type_truth in confusion_dict_doc:
                if type_truth not in confusion_dict_dataset:
                    confusion_dict_dataset[type_truth] = dict()

                for type_predicted in confusion_dict_doc[type_truth]:
                    if type_predicted not in confusion_dict_dataset[type_truth]:
                        confusion_dict_dataset[type_truth][type_predicted] = 0

                    confusion_dict_dataset[type_truth][type_predicted] += confusion_dict_doc[type_truth][type_predicted]

            n_files += 1

        print("\n\n------------------------ Summary ---------------------")
        print("Failed tables: {}".format(failed_tables_dataset))
        print("Processed {} files :: tables: {} :: statements: {} (with column(s) matched: {})".format(
            n_files, n_tables_dataset, n_statements_dataset, n_statements_with_column_matched_dataset))
        print("statements count frequency map")
        for n_statements in sorted(n_statements_table_frequency_map_dataset.keys()):
            print("\t{}: {} tables".format(n_statements, n_statements_table_frequency_map_dataset[n_statements]))
        print("Confusion dict:")
        print(confusion_dict_dataset)

        if not os.path.exists(args.statistics_dir):
            os.makedirs(args.statistics_dir)
        table_statistics_csv = os.path.join(args.statistics_dir, "table_statistics_{}.csv".format(data_split))
        pd.DataFrame(data=table_statistics_dataset).to_csv(path_or_buf=table_statistics_csv, index=False)
        if False:
            with io.open(file=table_statistics_csv, mode="wb") as fd:
                pd.DataFrame(data=table_statistics_dataset).to_csv(path_or_buf=fd, index=False)

def main(args):
    nlp_process_obj = NLPProcess()
    nlp_process_obj.load_nlp_model(True)
    dataset_obj = Dataset(nlp_process_obj=nlp_process_obj)
    if args.submit_dir == "None":
        submit_dir = None
    else:
        submit_dir = args.submit_dir
    dataset_obj.process_data_dir(data_dir=args.data_dir, data_split=args.data_split, flag_cell_span=args.flag_cell_span,
                                 flag_approx_string_match=args.flag_approx_string_match, submit_dir=submit_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="store",
                        default="C:/KA/data/NLP/statement_verification_evidence_finding/train_manual_v1.3.2/v1.3.2/ref/", dest="data_dir")
    parser.add_argument("--data_split", action="store", default="train", dest="data_split",
                        help="data split to be executed. Values permitted: train, dev, test")
    parser.add_argument("--flag_cell_span", action="store_true", default=False, dest="flag_cell_span",
                        help="bool to indicate whether row, col span is mentioned for each cell. data version 1.3 introduces span.")
    parser.add_argument("--flag_approx_string_match", action="store_true", default=False, dest="flag_approx_string_match")
    parser.add_argument("--submit_dir", action="store", default=os.path.join(os.path.dirname(__file__), "../output/res"), dest="submit_dir")
    parser.add_argument("--statistics_dir", action="store", default=os.path.join(os.path.dirname(__file__), "../output/statistics"), dest="statistics_dir")
    args = parser.parse_args()

    print("args: {}".format(args))
    main(args=args)
