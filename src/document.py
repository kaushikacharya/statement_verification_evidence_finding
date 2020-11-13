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

from src.table import Table

pd.set_option("display.max_columns", 20)


def create_valid_xml(xml_text):
    """Convert xml into a valid xml.
        N.B. This refers to data version: v1. In the next version v1.2, data is a valid xml format.
        XML provided as part of Shared Task crashes while parsing by ElementTree.
        Following two changes are done:
        1. <Table table_id> => <Table id="table_id">
        2. Given xml is put under the item Document.
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
    def __init__(self):
        pass

    def parse_xml(self, xml_file, table_tag="table", flag_modify_xml=False, verbose=False):
        """Parse xml of the document.

            Parameters
            ----------
            xml_file : filepath (XML document file path)
            table_tag : str (For v1 its "Table")
            flag_modify_xml : bool (True for data version v1)
            verbose : bool
        """
        assert os.path.exists(xml_file), "XML file; {} NOT found".format(xml_file)
        # tree = ET.parse(xml_file)
        # root = tree.getroot()
        with io.open(xml_file, encoding="utf-8") as fd:
            xml_text = fd.read()

        if flag_modify_xml:
            xml_text = create_valid_xml(xml_text=xml_text)

        root = ET.fromstring(xml_text)
        if True:
            print("root.tag: {} :: root.attrib: {}".format(root.tag, root.attrib))

        failed_tables = []
        n_statements_doc = 0
        n_statements_with_column_matched_doc = 0
        n_tables_doc = 0

        # iterate over the tables
        for table_item in root.findall(table_tag):
            if True:
                print("\n{} : {}".format(table_item.tag, table_item.attrib))

            try:
                table_obj = Table()
                table_obj.parse_xml(table_item=table_item, verbose=False)
                table_obj.process_table()
                n_statements_doc += len(table_obj.statements)
                n_tables_doc += 1
                for stmnt_i in range(len(table_obj.statements)):
                    if table_obj.statements[stmnt_i].columns_matched:
                        n_statements_with_column_matched_doc += 1
            except Exception:
                print("Failed in table id: {} :: file: {}".format(table_item.attrib["id"], xml_file))
                traceback.print_exc()
                failed_tables.append(table_item.attrib["id"])

        return failed_tables, n_tables_doc, n_statements_doc, n_statements_with_column_matched_doc


def main(args):
    xml_file = os.path.join(args.data_dir, args.filename)
    doc_obj = Document()
    doc_obj.parse_xml(xml_file=xml_file, verbose=args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="store", default="C:/KA/data/NLP/statement_verification_evidence_finding/v1.2/output/", dest="data_dir")
    parser.add_argument("--filename", action="store", dest="filename")
    parser.add_argument("--verbose", action="store_true", default=False, dest="verbose")
    args = parser.parse_args()

    print("args: {}".format(args))
    main(args=args)