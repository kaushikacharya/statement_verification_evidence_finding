#!/usr/bin/env python

import argparse
import io
import os
import re
import xml.etree.ElementTree as ET

"""
Process an XML document.
Document contains one or more tables.
"""

def create_valid_xml(xml_text):
    """Convert xml into a valid xml.
        XML provided as part of Shared Task crashes while parsing by ElementTree.
        Following two changes are done:
        1. <Table table_id> => <Table id="table_id">
        2. Given xml is put under the item TableSet.
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

    xml_text_mod = "<TableSet>/n"
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
    xml_text_mod += "</TableSet>"

    return xml_text_mod


class Document:
    """This class corresponds to the XML document of a paper.
        This contains table(s) and statements.
    """
    def __init__(self):
        pass

    def parse_xml(self, xml_file, flag_modify_xml=True, verbose=False):
        assert os.path.exists(xml_file), "XML file; {} NOT found".format(xml_file)
        # tree = ET.parse(xml_file)
        # root = tree.getroot()
        with io.open(xml_file, encoding="utf-8") as fd:
            xml_text = fd.read()

        if flag_modify_xml:
            xml_text = create_valid_xml(xml_text=xml_text)

        root = ET.fromstring(xml_text)
        if verbose:
            print("root.tag: {} :: root.attrib: {}".format(root.tag, root.attrib))

        for table_item in root.findall("Table"):
            if verbose:
                print("\n{} : {}".format(table_item.tag, table_item.attrib))
            for row in table_item.findall("row"):
                if verbose:
                    print("row: {}".format(row.attrib["row"]))
                for cell in row.findall("cell"):
                    # Comparing with accompanied html shows additional whitespaces, newlines have been added into cell text of the xml.
                    cell_text = re.sub(r"\s+", " ", cell.attrib["text"])
                    if verbose:
                        print("\tcol: {} :: text: {}".format(cell.attrib["col"], cell_text))

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
    parser.add_argument("--data_dir", action="store", default="C:/KA/data/NLP/statement_verification_evidence_finding/v1/output/", dest="data_dir")
    parser.add_argument("--filename", action="store", dest="filename")
    parser.add_argument("--verbose", action="store_true", default=False, dest="verbose")
    args = parser.parse_args()

    print("args: {}".format(args))
    main(args=args)