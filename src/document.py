#!/usr/bin/env python

"""
Process an XML document.
Document contains one or more tables.
"""

import argparse
import io
import os
import pandas as pd
import re
import traceback
import xml.etree.ElementTree as ET

from src.nlp_process import NLPProcess
from src.table import Table
from src.utils import *

pd.set_option("display.max_columns", 20)
# pd.set_option("display.encoding", "utf-8")


def create_valid_xml(xml_text):
    """Convert xml into a valid xml.
        N.B. This refers to data version: v1. In the next version v1.1, data is in valid xml format.
        XML provided as part of Shared Task crashes while parsing by ElementTree.
        Following two changes are done:
        1. <Table table_id> => <Table id="table_id">
        2. Given xml is put under the xml element Document.
    """
    table_pos_arr = []

    """
    Examples of <table_id>:
        <Table 1>
        <Table A1>
        <Table A.1>
    """
    for m in re.finditer(r"<[/]*Table [\w\.]+>", xml_text):
        if xml_text[m.start() + 1] == "/":
            table_id_start_pos = m.start() + len("</Table ")
        else:
            table_id_start_pos = m.start() + len("<Table ")
        table_id_end_pos = m.end() - 1
        if xml_text[m.start() + 1] == "/":
            table_item_mod = '</Table>'
        else:
            table_item_mod = '<Table id="' + xml_text[table_id_start_pos: table_id_end_pos] + '">'

        # store the text span positions in original text and the modified text
        table_pos_arr.append((m.start(), m.end(), table_item_mod))

    xml_text_mod = "<Document>/n"
    pos = 0
    for pos_i in range(len(table_pos_arr)):
        # text portion from the end of previous Table element
        # In case of first Table, take text portion from the start
        xml_text_mod += xml_text[pos:table_pos_arr[pos_i][0]]
        # modify the Table element
        xml_text_mod += table_pos_arr[pos_i][2]

        # update pos to the end of the current Table element
        pos = table_pos_arr[pos_i][1]

    # appending the rest of the text i.e. the text after the final Table element
    xml_text_mod += xml_text[pos:]
    xml_text_mod += "</Document>"

    return xml_text_mod


class Document:
    """This class corresponds to the XML document of a paper.
        This contains table(s) and statements.
    """
    def __init__(self, nlp_process_obj):
        self.doc_id = None
        self.nlp_process_obj = nlp_process_obj

    def parse_xml(self, xml_file, table_tag="table", table_id=None, flag_modify_xml=False, flag_cell_span=True, flag_approx_string_match=True, verbose=False):
        """Parse xml of the document.

            Parameters
            ----------
            xml_file : filepath (XML document file path)
            table_tag : str (For v1 its "Table")
            table_id : str (default: None, pass a value to execute a particular table)
            flag_modify_xml : bool (True only for data version v1)
            flag_cell_span : bool (True from data version 1.3 onwards)
            flag_approx_string_match : bool
            verbose : bool
        """
        assert os.path.exists(xml_file), "XML file; {} NOT found".format(xml_file)

        self.doc_id = os.path.splitext(os.path.basename(xml_file))[0]

        # tree = ET.parse(xml_file)
        # root = tree.getroot()
        with io.open(xml_file, encoding="utf-8") as fd:
            xml_text = fd.read()

        if flag_modify_xml:
            xml_text = create_valid_xml(xml_text=xml_text)

        # Adding xml declaration as a first line
        xml_text = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_text

        root = ET.fromstring(xml_text)
        if False:
            print("root.tag: {} :: root.attrib: {}".format(root.tag, root.attrib))

        failed_tables = []
        table_statistics = dict()
        table_predict = dict()
        n_statements_doc = 0
        n_statements_with_column_matched_doc = 0
        n_tables_doc = 0
        # maps frequency of tables for statements count
        # i.e. for key "x", value corresponding to it mentions how many tables are there with "x" statements
        n_statements_table_frequency_map_doc = dict()
        confusion_dict_doc = dict()

        doc_output_elem = ET.Element("document")
        summary_doc = list()

        # iterate over the tables
        for table_item in root.findall(table_tag):
            if True:
                print("\n{} : {}".format(table_item.tag, table_item.attrib))

            if table_id is not None and table_item.attrib["id"] != table_id:
                    continue

            try:
                table_obj = Table(doc_id=self.doc_id, nlp_process_obj=self.nlp_process_obj)
                table_obj.parse_xml(table_item=table_item, flag_cell_span=flag_cell_span, verbose=verbose)
                statement_id_predict_info_map = table_obj.process_table(flag_approx_string_match=flag_approx_string_match, verbose=verbose)
                n_statements_table = len(table_obj.statements)
                # Build the table element for submit
                table_output_elem = table_obj.build_table_element(ref_table_elem=table_item, statement_id_predict_info_map=statement_id_predict_info_map)
                # Append the table element into document element
                doc_output_elem.append(table_output_elem)
                n_statements_doc += n_statements_table
                n_tables_doc += 1

                table_item_id = table_item.attrib["id"]

                table_statistics[table_item_id] = dict()
                table_statistics[table_item_id]["n_column_rows"] = table_obj.table_data_start_row_index
                table_statistics[table_item_id]["n_data_rows"] = table_obj.table_data_end_row_index - table_obj.table_data_start_row_index

                table_predict[table_item_id] = dict()
                table_predict[table_item_id]["predict_info_map"] = statement_id_predict_info_map

                if n_statements_table not in n_statements_table_frequency_map_doc:
                    n_statements_table_frequency_map_doc[n_statements_table] = 0

                n_statements_table_frequency_map_doc[n_statements_table] += 1

                summary_table = dict()
                for stmnt_i in range(n_statements_table):
                    summary_table[table_obj.statements[stmnt_i].id] = \
                        {'text': table_obj.statements[stmnt_i].text, 'type_ground_truth': table_obj.statements[stmnt_i].type}
                    if table_obj.statements[stmnt_i].columns_matched:
                        n_statements_with_column_matched_doc += 1

                    type_truth = table_obj.statements[stmnt_i].type
                    if type_truth not in confusion_dict_doc:
                        confusion_dict_doc[type_truth] = dict()

                    stmnt_id = table_obj.statements[stmnt_i].id
                    assert stmnt_id in statement_id_predict_info_map, "predicted type missing for statement id: {}".format(stmnt_id)
                    type_predicted = statement_id_predict_info_map[stmnt_id]["type_predict"] if "type_predict" in statement_id_predict_info_map[stmnt_id] else None
                    if type_predicted is None:
                        type_predicted = "unknown"

                    if type_predicted not in confusion_dict_doc[type_truth]:
                        confusion_dict_doc[type_truth][type_predicted] = 0

                    confusion_dict_doc[type_truth][type_predicted] += 1

                for statements_elem in table_output_elem.findall('statements'):
                    for statement_elem in statements_elem.findall('statement'):
                        summary_table[statement_elem.get('id')]['type_predicted'] = statement_elem.get('type')

                summary_doc.append({'table_id': table_obj.table_id, 'summary_table': summary_table})

            except Exception:
                print("Failed in table id: {} :: file: {}".format(table_item.attrib["id"], xml_file))
                traceback.print_exc()
                failed_tables.append(table_item.attrib["id"])

        if verbose:
            print("\n------------------Summary--------------------")
            for table_i in range(len(summary_doc)):
                summary_table = summary_doc[table_i]['summary_table']
                print("\nTable: {} :: #statements: {}".format(summary_doc[table_i]['table_id'], len(summary_table)))
                # display erroneous predictions
                for statement_id in summary_table:
                    if summary_table[statement_id]['type_ground_truth'] is not None and \
                                    summary_table[statement_id]['type_ground_truth'] != summary_table[statement_id]['type_predicted']:
                        print("\tid: {} :: text: {}".format(statement_id, summary_table[statement_id]['text']))
                        print("\t\ttype: ground truth: {} :: predicted: {}".format(summary_table[statement_id]['type_ground_truth'], summary_table[statement_id]['type_predicted']))

        doc_output_tree = ET.ElementTree(doc_output_elem)

        output_dict = dict()
        output_dict["failed_tables_doc"] = failed_tables
        output_dict["n_tables_doc"] = n_tables_doc
        output_dict["n_statements_doc"] = n_statements_doc
        output_dict["n_statements_table_frequency_map_doc"] = n_statements_table_frequency_map_doc
        output_dict["n_statements_with_column_matched_doc"] = n_statements_with_column_matched_doc
        output_dict["table_statistics_doc"] = table_statistics
        output_dict["table_predict_doc"] = table_predict
        output_dict["doc_output_tree"] = doc_output_tree
        output_dict["confusion_dict_doc"] = confusion_dict_doc

        return output_dict


def main(args):
    xml_file = os.path.join(args.data_dir, args.filename)
    nlp_process_obj = NLPProcess()
    nlp_process_obj.load_nlp_model(verbose=args.verbose)
    doc_obj = Document(nlp_process_obj=nlp_process_obj)
    output_dict = doc_obj.parse_xml(xml_file=xml_file, table_id=args.table_id, flag_cell_span=args.flag_cell_span,
                                    flag_approx_string_match=args.flag_approx_string_match, verbose=args.verbose)

    submit_dir = os.path.join(os.path.dirname(__file__), "../output/res")
    if not os.path.exists(submit_dir):
        os.makedirs(submit_dir)

    doc_output_tree = output_dict["doc_output_tree"]
    if args.flag_prettify:
        doc_output_tree = prettify(doc_output_tree)
    submit_filepath = os.path.join(submit_dir, args.filename)
    with io.open(file=submit_filepath, mode="wb") as fd:
        doc_output_tree.write(fd)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="store", default="C:/KA/data/NLP/statement_verification_evidence_finding/train_manual_v1.3.2/v1.3.2/ref/", dest="data_dir")
    parser.add_argument("--filename", action="store", dest="filename")
    parser.add_argument("--table_id", action="store", default=None, dest="table_id", help="Process specified table only.")
    parser.add_argument("--flag_cell_span", action="store_true", default=False, dest="flag_cell_span",
                        help="bool to indicate whether row, col span is mentioned for each cell. data version 1.3 introduces span.")
    parser.add_argument("--flag_approx_string_match", action="store_true", default=False,
                        dest="flag_approx_string_match")
    parser.add_argument("--flag_prettify", action="store_true", default=False, dest="flag_prettify", help="pretty print XML")
    parser.add_argument("--verbose", action="store_true", default=False, dest="verbose")
    args = parser.parse_args()

    print("args: {}".format(args))
    main(args=args)
