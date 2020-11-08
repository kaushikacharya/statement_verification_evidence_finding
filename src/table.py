#!/usr/bin/env python

import pandas as pd
import re

from src.utils import *

class Table:
    def __init__(self):
        pass

    def parse_xml(self, table_item, verbose=False):
        """Parse table item of xml

            Parameters
            ----------
            table_item : table item
            verbose : bool
        """
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

                    cell_data = float(cell_text) if is_number(cell_text) else cell_text
                    cur_col_id = int(cell.attrib["col"])
                    table_row_data[cur_col_id] = cell_data

                table_data.append(table_row_data)

        if len(table_header) == 1:
            table_df = pd.DataFrame(data=table_data, columns=table_header[0])
        else:
            table_df = pd.DataFrame(data=table_data, columns=pd.MultiIndex.from_tuples(list(zip(*table_header))))

        if True:
            print(table_df)
            print("Column dtypes:\n{}".format(table_df.dtypes))

        for statements in table_item.findall('statements'):
            for statement in statements:
                # print(statement.tag, type(statement.attrib))
                statement_id = statement.attrib["id"]
                statement_text = statement.attrib["text"]
                statement_type = statement.attrib["type"]
                if verbose:
                    print("Statement: id: {} :: type: {} :: text: {}".format(statement_id, statement_type,
                                                                             statement_text))

        return table_df
