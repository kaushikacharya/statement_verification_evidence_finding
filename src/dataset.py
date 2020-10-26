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
        failed_files = []
        n_success_files = 0
        for xml_file_path in glob.iglob(os.path.join(data_dir, "*.xml")):
            # print(xml_file_path)
            try:
                doc_obj = Document()
                doc_obj.parse_xml(xml_file=xml_file_path, verbose=False)
                n_success_files += 1
            except Exception:
                print("Failed in {}".format(xml_file_path))
                traceback.print_exc()
                failed_files.append(os.path.basename(xml_file_path))

        print("Failed files: {}".format(failed_files))
        print("Successfully processed {} files".format(n_success_files))

def main(args):
    dataset_obj = Dataset()
    dataset_obj.process_data_dir(data_dir=args.data_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="store",
                        default="C:/KA/data/NLP/statement_verification_evidence_finding/v1_autogt/output/", dest="data_dir")
    args = parser.parse_args()

    print("args: {}".format(args))
    main(args=args)
