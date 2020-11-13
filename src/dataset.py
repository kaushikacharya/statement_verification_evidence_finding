#!/usr/bin/env python

import argparse
import glob
import io
import os
import traceback

from src.document import Document

class Dataset:
    def __init__(self):
        pass

    def process_data_dir(self, data_dir):
        failed_tables_dataset = []
        n_tables_dataset = 0
        n_statements_dataset = 0
        n_statements_with_column_matched_dataset = 0
        n_files = 0
        for xml_file_path in glob.iglob(os.path.join(data_dir, "*.xml")):
            print("\n{}".format(xml_file_path))
            doc_obj = Document()
            failed_tables_doc, n_tables_doc, n_statements_doc, n_statements_with_column_matched_doc = doc_obj.parse_xml(xml_file=xml_file_path, verbose=False)
            if len(failed_tables_doc) > 0:
                failed_tables_dataset.append({os.path.basename(xml_file_path): failed_tables_doc})
            n_tables_dataset += n_tables_doc
            n_statements_dataset += n_statements_doc
            n_statements_with_column_matched_dataset += n_statements_with_column_matched_doc
            n_files += 1

        print("Failed tables: {}".format(failed_tables_dataset))
        print("Processed {} files :: tables: {} :: statements: {} (with column(s) matched: {})".format(n_files, n_tables_dataset, n_statements_dataset, n_statements_with_column_matched_dataset))

def main(args):
    dataset_obj = Dataset()
    dataset_obj.process_data_dir(data_dir=args.data_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="store",
                        default="C:/KA/data/NLP/statement_verification_evidence_finding/v1.2/output/", dest="data_dir")
    args = parser.parse_args()

    print("args: {}".format(args))
    main(args=args)
