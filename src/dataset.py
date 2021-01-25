#!/usr/bin/env python

"""
N.B. If executing in console, execute the following as pre-requisite:
export PYTHONIOENCODING=UTF-8

Command: python -u -m src.dataset --flag_cell_span > ./output/dataset/dataset_train.txt 2>&1
"""

import argparse
import glob
import io
import os
import traceback

from tqdm import tqdm

from src.document import Document
from src.nlp_process import NLPProcess

class Dataset:
    def __init__(self, nlp_process_obj):
        self.nlp_process_obj = nlp_process_obj

    def process_data_dir(self, data_dir, flag_cell_span=True, submit_dir=None):
        failed_tables_dataset = []
        n_tables_dataset = 0
        n_statements_dataset = 0
        n_statements_with_column_matched_dataset = 0
        n_files = 0

        if submit_dir and not os.path.exists(submit_dir):
            os.makedirs(submit_dir)

        for xml_file_path in tqdm(glob.glob(os.path.join(data_dir, "*.xml"))):
            print("\n{}".format(xml_file_path))
            doc_obj = Document(nlp_process_obj=self.nlp_process_obj)
            doc_output_dict = doc_obj.parse_xml(xml_file=xml_file_path, flag_cell_span=flag_cell_span, verbose=False)

            failed_tables_doc = doc_output_dict["failed_tables_doc"]
            n_tables_doc = doc_output_dict["n_tables_doc"]
            n_statements_doc = doc_output_dict["n_statements_doc"]
            n_statements_with_column_matched_doc = doc_output_dict["n_statements_with_column_matched_doc"]
            doc_output_tree = doc_output_dict["doc_output_tree"]

            if submit_dir:
                submit_filepath = os.path.join(submit_dir, os.path.basename(xml_file_path))
                with io.open(file=submit_filepath, mode="wb") as fd:
                    doc_output_tree.write(fd)

            if len(failed_tables_doc) > 0:
                failed_tables_dataset.append({os.path.basename(xml_file_path): failed_tables_doc})
            n_tables_dataset += n_tables_doc
            n_statements_dataset += n_statements_doc
            n_statements_with_column_matched_dataset += n_statements_with_column_matched_doc
            n_files += 1

        print("Failed tables: {}".format(failed_tables_dataset))
        print("Processed {} files :: tables: {} :: statements: {} (with column(s) matched: {})".format(n_files, n_tables_dataset, n_statements_dataset, n_statements_with_column_matched_dataset))

def main(args):
    nlp_process_obj = NLPProcess()
    nlp_process_obj.load_nlp_model(True)
    dataset_obj = Dataset(nlp_process_obj=nlp_process_obj)
    if args.submit_dir == "None":
        submit_dir = None
    else:
        submit_dir = args.submit_dir
    dataset_obj.process_data_dir(data_dir=args.data_dir, flag_cell_span=args.flag_cell_span, submit_dir=submit_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="store",
                        default="C:/KA/data/NLP/statement_verification_evidence_finding/train_manual_v1.3.2/v1.3.2/ref/", dest="data_dir")
    parser.add_argument("--flag_cell_span", action="store_true", default=False, dest="flag_cell_span",
                        help="bool to indicate whether row, col span is mentioned for each cell. data version 1.3 introduces span.")
    parser.add_argument("--submit_dir", action="store", default=os.path.join(os.path.dirname(__file__), "../output/res"), dest="submit_dir")
    args = parser.parse_args()

    print("args: {}".format(args))
    main(args=args)
