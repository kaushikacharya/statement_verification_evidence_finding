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
import xml.etree.ElementTree as ET

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

        # iterate over the tables
        for table_item in root.findall(table_tag):
            if True:
                print("\n{} : {}".format(table_item.tag, table_item.attrib))

            # iterate over the rows
            max_row_id = -1
            max_col_id = -1
            continuous_cols_upto_arr = []
            for row in table_item.findall("row"):
                if verbose:
                    print("row: {}".format(row.attrib["row"]))
                cur_row_id = int(row.attrib["row"])
                if cur_row_id > max_row_id:
                    max_row_id = cur_row_id

                # iterate over the cells of the row
                cur_row_continuous_cols_upto = -1
                for cell in row.findall("cell"):
                    # Comparing with accompanied html shows additional whitespaces, newlines have been added into cell text of the xml.
                    cell_text = re.sub(r"\s+", " ", cell.attrib["text"])
                    if verbose:
                        print("\tcol: {} :: text: {}".format(cell.attrib["col"], cell_text))
                    cur_col_id = int(cell.attrib["col"])
                    if cur_col_id > max_col_id:
                        max_col_id = cur_col_id

                    if (cell_text != "") and (cur_col_id == (cur_row_continuous_cols_upto+1)):
                        cur_row_continuous_cols_upto = cur_col_id

                continuous_cols_upto_arr.append(cur_row_continuous_cols_upto)

            # identify the row from which table data starts
            table_data_row_id = None
            for row_id in range(max_row_id):
                if continuous_cols_upto_arr[row_id] == max_col_id:
                    if row_id == 0:
                        table_data_row_id = row_id + 1
                    else:
                        table_data_row_id = row_id
                    break

            if True:
                print("\nCount: rows: {} :: cols: {}".format(max_row_id + 1, max_col_id + 1))
                print("Header row range: ({},{}) :: data row range: ({},{})".format(0, table_data_row_id,
                                                                                    table_data_row_id,
                                                                                    max_row_id + 1))
            # Populate table dataframe
            table_header = []
            table_data = []  # list of list
            for row in table_item.findall("row"):
                cur_row_id = int(row.attrib["row"])

                if cur_row_id < table_data_row_id:
                    # row represents column headers
                    prev_col_id = -1
                    prev_col_text = None
                    table_row_header = [None for i in range(max_col_id+1)]
                    for cell in row.findall("cell"):
                        cell_text = re.sub(r"\s+", " ", cell.attrib["text"])
                        cur_col_id = int(cell.attrib["col"])

                        if cell_text == "":
                            continue

                        # Assign previous column header to the column positions between previous cell and current cell
                        # Represents the case of nested headers. The current column will have sub-columns nested under it in the next row.
                        for col_id in range(prev_col_id+1, cur_col_id):
                            table_row_header[col_id] = prev_col_text

                        # Assign column header to current column position
                        table_row_header[cur_col_id] = cell_text

                        # update
                        prev_col_id = cur_col_id
                        prev_col_text = cell_text

                    # Assigning for the last columns in case of nested header
                    for col_id in range(prev_col_id+1, max_col_id+1):
                        table_row_header[col_id] = prev_col_text

                    table_header.append(table_row_header)
                else:
                    # row represents table data
                    table_row_data = [None for i in range(max_col_id+1)]
                    for cell in row.findall("cell"):
                        cell_text = re.sub(r"\s+", " ", cell.attrib["text"])
                        cur_col_id = int(cell.attrib["col"])
                        table_row_data[cur_col_id] = cell_text

                    table_data.append(table_row_data)

            if len(table_header) == 1:
                table_df = pd.DataFrame(data=table_data, columns=table_header[0])
            else:
                table_df = pd.DataFrame(data=table_data, columns=pd.MultiIndex.from_tuples(list(zip(*table_header))))

            if True:
                print(table_df)

            for statements in table_item.findall('statements'):
                for statement in statements:
                    # print(statement.tag, type(statement.attrib))
                    statement_id = statement.attrib["id"]
                    statement_text = statement.attrib["text"]
                    statement_type = statement.attrib["type"]
                    if verbose:
                        print("Statement: id: {} :: type: {} :: text: {}".format(statement_id, statement_type,
                                                                                 statement_text))


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