#!/usr/bin/env python

import os
import pandas as pd
import re

from pandas.api.types import is_numeric_dtype
from spacy import displacy

from src.utils import *

# pd.set_option("display.max_columns", 20)
# pd.set_option("display.encoding", "utf-8")


class Statement:
    def __init__(self, statement_id, text, statement_type=None, columns_matched=None):
        self.id = statement_id
        self.text = text
        self.type = statement_type
        self.columns_matched = columns_matched


class Table:
    def __init__(self, doc_id, nlp_process_obj):
        self.doc_id = doc_id
        self.nlp_process_obj = nlp_process_obj
        self.table_id = None
        self.caption_text = ""
        self.legend_text = ""
        self.df = None
        self.statements = []

    def parse_xml(self, table_item, verbose=False):
        """Parse table element of xml

            Parameters
            ----------
            table_item : table element
            verbose : bool
        """

        self.table_id = table_item.attrib["id"]

        # extract caption text
        caption = table_item.find('caption')
        if caption:
            self.caption_text = re.sub(r"\s+", " ", caption.attrib['text'])

        # extract legend text
        legend = table_item.find('legend')
        if legend:
            self.legend_text = re.sub(r"\s+", " ", legend.attrib['text'])

        # iterate over the rows to identify column range and row range
        # This helps in identifying empty columns at the end which can be skipped while forming the dataframe
        max_row_id = -1
        max_col_id = -1
        min_col_id = None  # represents the start column of headers

        for row in table_item.findall("row"):
            if verbose:
                print("row: {}".format(row.attrib["row"]))
            cur_row_id = int(row.attrib["row"])
            if cur_row_id > max_row_id:
                max_row_id = cur_row_id

            # iterate over the cells of the row
            for cell in row.findall("cell"):
                # Comparing with accompanied html shows additional whitespaces, newlines have been added into cell text of the xml.
                cell_text = re.sub(r"\s+", " ", cell.attrib["text"])
                if verbose:
                    print("\tcol: {} :: text: {}".format(cell.attrib["col"], cell_text))
                if cell_text == "":
                    # consider empty text cells as None
                    # e.g. 20661.xml : Last 3 columns have empty text
                    continue
                cur_col_id = int(cell.attrib["col"])
                if cur_col_id > max_col_id:
                    max_col_id = cur_col_id
                if min_col_id is None:
                    min_col_id = cur_col_id

        # identify the row from which table data starts
        # TODO Handle the table with row index. In this case column header is absent corresponding to the column mentioning row index.
        table_data_row_id = None
        empty_col_set = set(range(min_col_id, max_col_id + 1))
        for i in range(2):
            for row in table_item.findall("row"):
                cur_row_id = int(row.attrib["row"])

                for cell in row.findall("cell"):
                    cell_text = re.sub(r"\s+", " ", cell.attrib["text"])
                    if cell_text == "":
                        continue
                    cur_col_id = int(cell.attrib["col"])
                    if cur_col_id in empty_col_set:
                        empty_col_set.remove(cur_col_id)

                if len(empty_col_set) == 0:
                    table_data_row_id = cur_row_id + 1
                    break

            if len(empty_col_set) == 0:
                # case: Usual cases where there's no empty column(s)
                break
            else:
                # case: Empty columns in-between columns with data filled
                #   Re-process above code segment to identify from which row table data starts
                #   e.g. document# 20690
                # TODO Store these columns as they might be useful in identifying nested column headers
                # TODO Also these empty columns could be a signal of range for nested column headers.
                empty_col_set = set(range(min_col_id, max_col_id + 1)).difference(empty_col_set)

        assert table_data_row_id is not None, "table_data_row_id not set"
        assert table_data_row_id <= max_row_id, "table_data_row_id: {} > max_row_id: {}".format(table_data_row_id,
                                                                                                max_row_id)

        if verbose:
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
                table_row_header = [None for i in range(max_col_id + 1)]
                for cell in row.findall("cell"):
                    cell_text = re.sub(r"\s+", " ", cell.attrib["text"])
                    if cell_text == "":
                        continue

                    cur_col_id = int(cell.attrib["col"])

                    # Assign previous column header to the column positions between previous cell and current cell
                    # Represents the case of nested headers. The current column will have sub-columns nested under it in the next row.
                    for col_id in range(prev_col_id + 1, cur_col_id):
                        table_row_header[col_id] = prev_col_text

                    # Assign column header to current column position
                    table_row_header[cur_col_id] = cell_text

                    # update
                    prev_col_id = cur_col_id
                    prev_col_text = cell_text

                # Assigning for the last columns in case of nested header
                for col_id in range(prev_col_id + 1, max_col_id + 1):
                    table_row_header[col_id] = prev_col_text

                table_header.append(table_row_header)
            else:
                # row represents table data
                table_row_data = [None for i in range(max_col_id + 1)]
                for cell in row.findall("cell"):
                    cell_text = re.sub(r"\s+", " ", cell.attrib["text"])
                    if cell_text == "":
                        continue

                    _, cell_data = is_number(cell_text)
                    # cell_data = float(cell_text) if is_number(cell_text) else cell_text
                    cur_col_id = int(cell.attrib["col"])
                    table_row_data[cur_col_id] = cell_data

                table_data.append(table_row_data)

        if len(table_header) == 1:
            table_df = pd.DataFrame(data=table_data, columns=table_header[0])
        else:
            table_df = pd.DataFrame(data=table_data, columns=pd.MultiIndex.from_tuples(list(zip(*table_header))))

        self.df = table_df

        if True:
            print(table_df)
            print("Column dtypes:\n{}".format(table_df.dtypes))
            # Identify columns having numeric data type
            numeric_columns = []
            for col_name in table_df.columns:
                if is_numeric_dtype(table_df[col_name]):
                    numeric_columns.append(col_name)
            if len(numeric_columns):
                print("Numeric columns: {}".format(numeric_columns))

        if verbose:
            m = re.search(r'\d+', self.table_id)
            assert m, "Table id: {} does not contain numerals".format(self.table_id)
            output_dir = os.path.join(os.path.dirname(__file__), "../output/debug", self.doc_id, m.group(0))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        n_statements = 0
        for statements in table_item.findall('statements'):
            for statement in statements:
                n_statements += 1
                # print(statement.tag, type(statement.attrib))
                statement_id = statement.attrib["id"]
                statement_text = statement.attrib["text"]
                statement_type = statement.attrib["type"]
                doc_statement = self.nlp_process_obj.construct_doc(text=statement_text)

                if True:
                    print("Statement: id: {} :: type: {} :: text: {}".format(statement_id, statement_type,
                                                                             statement_text))

                if verbose:
                    svg = displacy.render(doc_statement, style="dep")
                    svg_statement_file = os.path.join(output_dir, statement_id + ".svg")
                    with open(svg_statement_file, mode="w", encoding="utf-8") as fd:
                        fd.write(svg)

                if statement_type == "":
                    statement_type = None

                self.statements.append(Statement(statement_id=statement_id, text=statement_text, statement_type=statement_type))

    def process_table(self, verbose=False):
        if isinstance(self.df.columns, pd.MultiIndex):
            return
        column_names = [x for x in self.df.columns if x]
        if len(column_names) == 0:
            return

        # regex pattern to search for column names
        # N.B. Fails when words have alphanumeric characters. e.g. Column: Pmid [m]   20856.xml
        if False:
            regex_pattern = r'('
            for i, column_name in enumerate(column_names):
                if i > 0:
                    regex_pattern += r'|'
                regex_pattern += r'\b{}\b'.format(column_name)
            regex_pattern += r')'

        for stmnt_i in range(len(self.statements)):
            print("Statement #{} :: id: {}".format(stmnt_i, self.statements[stmnt_i].id))
            # Identify columns in statement
            if False:
                columns_matched = re.findall(pattern=regex_pattern, string=self.statements[stmnt_i].text, flags=re.I)
                if len(columns_matched) > 0:
                    print("\tColumns matched: {}".format(columns_matched))

            statement_text_lower = self.statements[stmnt_i].text.lower()
            columns_matched = []
            for column_name in column_names:
                start_pos_arr = [i for i in range(len(statement_text_lower)) if statement_text_lower.startswith(column_name.lower(), i)]
                flag_column_matched = False
                for start_pos in start_pos_arr:
                    if start_pos > 0 and statement_text_lower[start_pos-1] != " ":
                        continue
                    end_pos = start_pos + len(column_name)
                    if end_pos < len(statement_text_lower) and statement_text_lower[end_pos] != " ":
                        continue
                    flag_column_matched = True

                if flag_column_matched:
                    columns_matched.append(column_name)
                    print("\tColumn matched: {}".format(column_name))

            if columns_matched:
                self.statements[stmnt_i].columns_matched = columns_matched

            m = re.search(r'(\bhighest\b|\bgreatest\b|\blowest\b)', self.statements[stmnt_i].text, flags=re.I)
            if m:
                print("\t{}".format(m.group(0)))
