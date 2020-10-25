#!/usr/bin/env python

import argparse
import io
import os
import re
import xml.etree.ElementTree as ET


class Table:
    def __init__(self):
        pass

    def parse_xml(self):
        pass


def main(args):
    xmlfile = os.path.join(args.data_dir, args.filename)
    with io.open(xmlfile, encoding="utf-8") as fd:
        # tree = ET.parse(xmlfile)
        # root = tree.getroot()
        xml_text = fd.read()
        table_pos_arr = []

        for m in re.finditer(r"<[/]*Table \d+>", xml_text):
            if xml_text[m.start()+1] == "/":
                table_id_start_pos = m.start() + len("</Table ")
            else:
                table_id_start_pos = m.start() + len("<Table ")
            table_id_end_pos = m.end() - 1
            if xml_text[m.start() + 1] == "/":
                table_item_mod = '</Table>'
            else:
                table_item_mod = '<Table id="' + xml_text[table_id_start_pos: table_id_end_pos] + '">'
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

        root = ET.fromstring(xml_text_mod)
        print("root.tag: {} :: root.attrib: {}".format(root.tag, root.attrib))

        for table_item in root.findall("Table"):
            print("\n{} : {}".format(table_item.tag, table_item.attrib))
            for row in table_item.findall("row"):
                print("row: {}".format(row.attrib["row"]))
                for cell in row.findall("cell"):
                    print("\tcol: {} :: text: {}".format(cell.attrib["col"], cell.attrib["text"]))

            for statements in table_item.findall('statements'):
                for statement in statements:
                    # print(statement.tag, type(statement.attrib))
                    statement_id = statement.attrib["id"]
                    statement_text = statement.attrib["text"]
                    statement_type = statement.attrib["type"]
                    print("Statement: id: {} :: type: {} :: text: {}".format(statement_id, statement_type, statement_text))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="store", default="C:/KA/data/NLP/statement_verification_evidence_finding/v1/output/", dest="data_dir")
    parser.add_argument("--filename", action="store", dest="filename")
    args = parser.parse_args()

    print("args: {}".format(args))
    main(args=args)

