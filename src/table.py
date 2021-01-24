#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import re
import xml.etree.ElementTree as ET

from pandas.api.types import is_numeric_dtype
from spacy import displacy
from spacy.tokenizer import Tokenizer
from word2number import w2n

from src.utils import *

# pd.set_option("display.max_columns", 20)
# pd.set_option("display.encoding", "utf-8")


class Token:
    def __init__(self, text, lemma=None, normalized_text=None, part_of_speech_coarse=None,
                 part_of_speech_fine_grained=None, dependency_tag=None, head_index=None, children_index_arr=None):
        self.text = text
        self.lemma = lemma
        self.normalized_text = normalized_text
        self.part_of_speech_coarse = part_of_speech_coarse
        self.part_of_speech_fine_grained = part_of_speech_fine_grained
        self.dependency_tag = dependency_tag
        self.head_index = head_index
        self.children_index_arr = children_index_arr


class Statement:
    def __init__(self, statement_id, text, statement_type=None, columns_matched=None, tokens=None):
        self.id = statement_id
        self.text = text
        self.type = statement_type
        self.columns_matched = columns_matched
        self.tokens = tokens


class Table:
    def __init__(self, doc_id, nlp_process_obj):
        self.doc_id = doc_id
        self.nlp_process_obj = nlp_process_obj
        self.table_id = None
        self.caption_text = ""
        self.legend_text = ""
        self.table_data_start_row_index = None
        self.table_data_end_row_index = None
        self.table_data_start_col_index = 0
        # Though `df` currently stores columns starting from 0, but column headers prior to `table_min_col_index' are None.
        self.table_min_col_index = None
        # dataframe to store table.
        #   Table row indices in range (0, self.table_data_start_row_index) represents column headers.
        #   Table row indices in range (self.table_data_start_row_index, self.table_data_end_row_index) represents values.
        self.df = None
        # dict of dict: keys: [row_index][col_index],  values: list of Token
        self.cell_info_dict = dict()
        # dict of dict: keys: [row_index][col_index],  values: set of statement ids for which the cell is relevant
        self.cell_evidence_dict = dict()
        self.statements = []

    def parse_xml(self, table_item, flag_cell_span=True, verbose=False):
        """Parse table element of xml

            Parameters
            ----------
            table_item : table element
            flag_cell_span : bool
            verbose : bool
        """

        self.table_id = table_item.attrib["id"]

        # extract caption text
        caption = table_item.find('caption')
        if caption is not None:
            self.caption_text = re.sub(r"\s+", " ", caption.attrib['text'])

        if verbose and len(self.caption_text) > 0:
            print("Caption: {}".format(self.caption_text))

        # extract legend text
        legend = table_item.find('legend')
        if legend is not None:
            self.legend_text = re.sub(r"\s+", " ", legend.attrib['text'])

        if verbose and len(self.legend_text) > 0:
            print("Legend: {}".format(self.legend_text))

        # Iterate over the table rows to
        #   a) identify column range and row range
        #   b) populate cell_info_dict
        # This helps in identifying empty columns at the end which can be skipped while forming the dataframe
        max_row_id = -1
        max_col_id = -1
        min_col_id = None  # represents the start column of headers

        for row in table_item.iterfind("row"):
            if verbose:
                print("row: {}".format(row.attrib["row"]))
            cur_row_id = int(row.attrib["row"])
            if cur_row_id > max_row_id:
                max_row_id = cur_row_id

            # iterate over the cells of the row
            for cell in row.iterfind("cell"):
                # Normalizing cell text by removing additional whitespaces, newlines.
                #  These can be observed by comparing with corresponding html.
                cell_text = re.sub(r"\s+", " ", cell.attrib["text"])
                cell_text = modify_text(cell_text)
                if verbose:
                    if flag_cell_span:
                        print("\tcol range(end inclusive): start: {} : end: {} :: row range(end inclusive): start: {} : end: {} :: text: {}".format(
                            cell.attrib["col-start"], cell.attrib["col-end"], cell.attrib["row-start"], cell.attrib["row-end"], cell_text))
                    else:
                        print("\tcol: {} :: text: {}".format(cell.attrib["col"], cell_text))
                if cell_text == "":
                    # consider empty text cells as None
                    # e.g. 20661.xml : Last 3 columns have empty text
                    continue

                if flag_cell_span:
                    cur_col_id = int(cell.attrib["col-start"])
                else:
                    cur_col_id = int(cell.attrib["col"])
                if cur_col_id > max_col_id:
                    max_col_id = cur_col_id
                if min_col_id is None:
                    min_col_id = cur_col_id

                if cur_row_id not in self.cell_info_dict:
                    self.cell_info_dict[cur_row_id] = dict()
                # insert empty list which will be populated with list of Token objects
                self.cell_info_dict[cur_row_id][cur_col_id] = []

                if cur_row_id not in self.cell_evidence_dict:
                    self.cell_evidence_dict[cur_row_id] = dict()
                # insert empty set which will be later populated with statement ids for which the cell is relevant
                self.cell_evidence_dict[cur_row_id][cur_col_id] = set()

                doc_cell = self.nlp_process_obj.construct_doc(text=cell_text)
                for token in doc_cell:
                    token_obj = Token(text=token.text, lemma=token.lemma_, normalized_text=token.norm_,
                                      part_of_speech_coarse=token.pos_, part_of_speech_fine_grained=token.tag_,
                                      dependency_tag=token.dep_, head_index=token.head.i,
                                      children_index_arr=[child.i for child in token.children])
                    self.cell_info_dict[cur_row_id][cur_col_id].append(token_obj)
                    if verbose:
                        print("\t\ttoken: #{} :: text: {} POS: {}".format(token.i, token.text, token.pos_))

        assert min_col_id is not None, "min_col_id is not assigned"
        self.table_min_col_index = min_col_id

        # Identify the row from which table data starts
        # The rows prior to that will populate column headers of `df`
        # TODO Handle the table with row index. In this case column header is absent corresponding to the column mentioning row index.
        table_data_row_id = None
        empty_col_set = set(range(min_col_id, max_col_id + 1))
        for i in range(2):
            for row in table_item.iterfind("row"):
                cur_row_id = int(row.attrib["row"])

                for cell in row.iterfind("cell"):
                    cell_text = re.sub(r"\s+", " ", cell.attrib["text"])
                    if cell_text == "":
                        continue
                    if flag_cell_span:
                        cur_col_id = int(cell.attrib["col-start"])
                    else:
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
                #   In 2nd iteration these internal empty columns are not processed for the purpose of identifying
                #       row from which table data starts.
                # TODO Store these columns as they might be useful in identifying nested column headers
                # TODO Also these empty columns could be a signal of range for nested column headers.
                empty_col_set = set(range(min_col_id, max_col_id + 1)).difference(empty_col_set)

        assert table_data_row_id is not None, "table_data_row_id not set"
        # Hack to avoid crash
        if table_data_row_id > max_row_id:
            table_data_row_id = max_row_id
        assert table_data_row_id <= max_row_id, "table_data_row_id: {} > max_row_id: {}".format(table_data_row_id,
                                                                                                max_row_id)
        self.table_data_start_row_index = table_data_row_id
        self.table_data_end_row_index = max_row_id + 1

        if verbose:
            print("\nCount: rows: {} :: cols: {}".format(max_row_id + 1, max_col_id + 1))
            print("Header row range: ({},{}) :: data row range: ({},{})".format(0, table_data_row_id, table_data_row_id, max_row_id + 1)) #noqa

        # Populate table dataframe
        # ?? Are we not going to use min_col_id
        table_header = []
        table_data = []  # list of list
        for row in table_item.iterfind("row"):
            cur_row_id = int(row.attrib["row"])

            if cur_row_id < table_data_row_id:
                # row represents column headers
                prev_col_id = -1
                prev_col_text = None
                table_row_header = [None for i in range(max_col_id + 1)]
                for cell in row.iterfind("cell"):
                    cell_text = re.sub(r"\s+", " ", cell.attrib["text"])
                    cell_text = modify_text(cell_text)
                    if cell_text == "":
                        continue

                    if flag_cell_span:
                        cur_col_id = int(cell.attrib["col-start"])
                    else:
                        cur_col_id = int(cell.attrib["col"])

                    # Assign previous column header to the column positions between previous cell and current cell
                    # Represents the case of nested headers. The current column will have sub-columns nested under it in the next row.
                    # TODO For data version having cell span, we can utilize the span to replace approach used for previous data versions.
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
                for cell in row.iterfind("cell"):
                    cell_text = re.sub(r"\s+", " ", cell.attrib["text"])
                    cell_text = modify_text(cell_text)
                    if cell_text == "":
                        continue

                    _, cell_data = is_number(cell_text)
                    # cell_data = float(cell_text) if is_number(cell_text) else cell_text
                    if flag_cell_span:
                        cur_col_id = int(cell.attrib["col-start"])
                    else:
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
            # N.B. It's better to refer columns by index rather than its name.
            #       In case of duplicate column names, it selects `DataFrame` instead of `Series`.
            #       This leads to either of
            #           a) TypeError: Level type mismatch: nan (in case on None column headers
            #           b) False return by is_numeric_dtype() since the function expects array like object and not dataframe.
            numeric_columns = []
            for col_i, col_name in enumerate(table_df.columns):
                if is_numeric_dtype(table_df.iloc[:, col_i]):
                    numeric_columns.append((col_i, col_name))
            if False:
                for col_name in table_df.columns:
                    if is_numeric_dtype(table_df[col_name]):
                        numeric_columns.append(col_name)
            if len(numeric_columns):
                print("Numeric columns: {}".format(numeric_columns))

        output_dir = None
        if verbose:
            m = re.search(r'\d+', self.table_id)
            assert m, "Table id: {} does not contain numerals".format(self.table_id)
            output_dir = os.path.join(os.path.dirname(__file__), "../output/debug", self.doc_id, m.group(0))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        n_statements = 0
        for statements in table_item.iterfind('statements'):
            for statement in statements:
                n_statements += 1
                # print(statement.tag, type(statement.attrib))
                statement_id = statement.attrib["id"]
                statement_text = statement.attrib["text"]
                statement_text = modify_text(statement_text)
                statement_type = statement.attrib["type"]
                doc_statement = self.nlp_process_obj.construct_doc(text=statement_text)

                if verbose:
                    print("Statement: id: {} :: type: {} :: text: {}".format(statement_id, statement_type,
                                                                             statement_text))

                statement_obj = Statement(statement_id=statement_id, text=statement_text, statement_type=statement_type)
                statement_obj.tokens = []
                for token in doc_statement:
                    token_obj = Token(text=token.text, lemma=token.lemma_, normalized_text=token.norm_,
                                      part_of_speech_coarse=token.pos_, part_of_speech_fine_grained=token.tag_,
                                      dependency_tag=token.dep_, head_index=token.head.i,
                                      children_index_arr=[child.i for child in token.children])
                    statement_obj.tokens.append(token_obj)

                if verbose:
                    for token in doc_statement:
                        print("\ttoken: #{} :: text: {} :: lemma: {} :: norm: {} :: POS: {} :: tag: {} :: dep: {}".format(
                            token.i, token.text, token.lemma_, token.norm_, token.pos_, token.tag_, token.dep_))

                if verbose:
                    svg = displacy.render(doc_statement, style="dep")
                    assert output_dir is not None, "output_dir not assigned"
                    svg_statement_file = os.path.join(output_dir, statement_id + ".svg")
                    with open(svg_statement_file, mode="w", encoding="utf-8") as fd:
                        fd.write(svg)

                if statement_type == "":
                    statement_type = None

                self.statements.append(statement_obj)

    def process_table(self, verbose=False):
        """Process table for statement verification"""
        if False:
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

        # key: statement id  value: predicted type
        statement_id_predict_type_map = dict()

        if verbose:
            print("\n")

        for stmnt_i in range(len(self.statements)):
            if verbose:
                print("Statement #{} :: id: {} :: type(ground truth): {} :: text: {}".format(
                    stmnt_i, self.statements[stmnt_i].id, self.statements[stmnt_i].type, self.statements[stmnt_i].text))

            # Identify columns in statement
            if False:
                columns_matched = re.findall(pattern=regex_pattern, string=self.statements[stmnt_i].text, flags=re.I)
                if len(columns_matched) > 0:
                    print("\tColumns matched: {}".format(columns_matched))

            if False:
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

            # iterate over the table cell corresponding to the column headers
            # columns_matched = []
            flag_col_matched_arr = [None for i in range(len(self.df.columns))]
            # TODO Introduce column matching score to allow partial matches
            column_matched_tokens_dict = dict()
            for row_index in range(self.table_data_start_row_index):
                if row_index not in self.cell_info_dict:
                    continue
                for col_index in range(len(self.df.columns)):
                    if col_index not in self.cell_info_dict[row_index]:
                        continue
                    # Identify if any portion of the column text can be treated as optional
                    #   e.g. 20509.xml, Table 1 :: Column: Concentration (mg l −1 )
                    #           measurement units can be considered as optional in the statement
                    optional_token_begin = None
                    if self.cell_info_dict[row_index][col_index][-1].part_of_speech_coarse == "PUNCT":
                        begin_optional_text = None
                        if self.cell_info_dict[row_index][col_index][-1].text == ")":
                            begin_optional_text = "("
                        elif self.cell_info_dict[row_index][col_index][-1].text == "]":
                            begin_optional_text = "["

                        if begin_optional_text:
                            token_i = 1
                            while token_i < len(self.cell_info_dict[row_index][col_index]):
                                if self.cell_info_dict[row_index][col_index][token_i].text == begin_optional_text:
                                    optional_token_begin = token_i
                                    break
                                token_i += 1

                    match_length = optional_token_begin if optional_token_begin else len(self.cell_info_dict[row_index][col_index])
                    cur_column_matched_tokens_dict = dict()
                    for token_index_stmnt in range(len(self.statements[stmnt_i].tokens) - match_length + 1):
                        token_i = 0
                        flag_col_cell_matched = True
                        while token_i < match_length:
                            if (self.statements[stmnt_i].tokens[token_index_stmnt + token_i].text.lower() !=
                                    self.cell_info_dict[row_index][col_index][token_i].text.lower()) and \
                                    (self.statements[stmnt_i].tokens[token_index_stmnt + token_i].lemma.lower() !=
                                         self.cell_info_dict[row_index][col_index][token_i].lemma.lower()):
                                flag_col_cell_matched = False
                                break
                            token_i += 1

                        if flag_col_cell_matched:
                            flag_col_cell_full_matched = True
                            if optional_token_begin:
                                # check if it matches full column name
                                if token_index_stmnt + len(self.cell_info_dict[row_index][col_index]) > len(self.statements[stmnt_i].tokens):
                                    flag_col_cell_full_matched = False
                                else:
                                    token_j = token_i + 1
                                    while token_j < len(self.cell_info_dict[row_index][col_index]):
                                        if (self.statements[stmnt_i].tokens[token_index_stmnt + token_j].text.lower() !=
                                                self.cell_info_dict[row_index][col_index][token_j].text.lower()) and \
                                                (self.statements[stmnt_i].tokens[token_index_stmnt + token_j].lemma.lower() !=
                                                     self.cell_info_dict[row_index][col_index][token_j].lemma.lower()):
                                            flag_col_cell_full_matched = False
                                            break
                                        token_j += 1

                                    if flag_col_cell_full_matched:
                                        token_i = token_j

                            if 'token_index_range_statement' not in cur_column_matched_tokens_dict:
                                cur_column_matched_tokens_dict['token_index_range_statement'] = []
                            cur_column_matched_tokens_dict['token_index_range_statement'].append((token_index_stmnt, token_index_stmnt+token_i))

                            col_info = (col_index, token_index_stmnt, token_index_stmnt+token_i)
                            # columns_matched.append(col_info)
                            col_cell_text = " ".join([x.text for x in self.cell_info_dict[row_index][col_index]])
                            if verbose:
                                print("\tColumn cell matched: name: {} :: col info: {}".format(col_cell_text, col_info))
                                if not flag_col_cell_full_matched:
                                    print("\t\tPartial matched")

                    if len(cur_column_matched_tokens_dict) > 0:
                        if flag_col_matched_arr[col_index] is False:
                            # Column failed to match in one of the previous column header row(s).
                            pass
                        else:
                            if flag_col_matched_arr[col_index] is None:
                                flag_col_matched_arr[col_index] = True

                            if col_index not in column_matched_tokens_dict:
                                column_matched_tokens_dict[col_index] = []
                            column_matched_tokens_dict[col_index].extend(cur_column_matched_tokens_dict['token_index_range_statement'])
                    else:
                        flag_col_matched_arr[col_index] = False
                        # remove if it was added in any of the previous column header row
                        if col_index in column_matched_tokens_dict:
                            column_matched_tokens_dict.pop(col_index)

            if len(column_matched_tokens_dict) > 1:
                # remove column(s) which are subset of another column in terms of statement tokens
                col_index_arr = [k for k, v in sorted(column_matched_tokens_dict.items(), key=lambda x: x[1])]
                prev_col_index = col_index_arr[0]
                for cur_col_index in col_index_arr[1:]:
                    if (len(column_matched_tokens_dict[cur_col_index]) == 1 and len(column_matched_tokens_dict[prev_col_index]) == 1) and \
                            (column_matched_tokens_dict[cur_col_index][0][0] < column_matched_tokens_dict[prev_col_index][0][1]) and \
                            (column_matched_tokens_dict[cur_col_index][0][1] <= column_matched_tokens_dict[prev_col_index][0][1]):
                        # Currently considering subset only for non-disjoint text span
                        flag_col_matched_arr[cur_col_index] = False
                        column_matched_tokens_dict.pop(cur_col_index)
                    else:
                        prev_col_index = cur_col_index

            if verbose:
                col_matched_arr = [self.df.columns[i] for i in range(len(flag_col_matched_arr)) if flag_col_matched_arr[i]]
                if len(col_matched_arr) > 0:
                    print("\tColumns matched: {}".format(col_matched_arr))

            # Match row values of the first column to the statement (if the column is non-numeric)
            # iterate over the table cell rows which correspond to the data i.e. excluding rows corresponding to the column headers
            col_index = 0
            rows_matched = []
            # ?? 0th column could be id's (written as int). Do we need to discriminate this with other numeric value?
            # if not is_numeric_dtype(self.df.iloc[:, col_index]):
            for row_index in range(self.table_data_start_row_index, self.table_data_end_row_index):
                # N.B. row_index refers to table row. For row of dataframe, we need to offset by table_data_start_row_index
                if row_index not in self.cell_info_dict:
                    continue
                if col_index not in self.cell_info_dict[row_index]:
                    continue
                for token_index_stmnt in range(len(self.statements[stmnt_i].tokens) - len(self.cell_info_dict[row_index][col_index]) + 1):
                    token_i = 0
                    flag_row_matched = True
                    while token_i < len(self.cell_info_dict[row_index][col_index]):
                        if self.statements[stmnt_i].tokens[token_index_stmnt+token_i].text.lower() != \
                                self.cell_info_dict[row_index][col_index][token_i].text.lower():
                            flag_row_matched = False
                            break
                        token_i += 1

                    if flag_row_matched:
                        row_info = (row_index, token_index_stmnt, token_index_stmnt+token_i)
                        rows_matched.append(row_info)
                        row_name = self.df.iloc[row_index-self.table_data_start_row_index, col_index]
                        # row_name = self.df[self.df.columns[0]][row_index-self.table_data_start_row_index]
                        if verbose:
                            print("\tRow matched: name: {} :: row info: {}".format(row_name, row_info))

            if False:
                for row_name in self.df[column_names[0]]:
                    start_pos_arr = [i for i in range(len(statement_text_lower)) if
                                     statement_text_lower.startswith(row_name.lower(), i)]
                    flag_row_matched = False
                    for start_pos in start_pos_arr:
                        if start_pos > 0 and statement_text_lower[start_pos - 1] != " ":
                            continue
                        end_pos = start_pos + len(row_name)
                        if end_pos < len(statement_text_lower) and statement_text_lower[end_pos] != " ":
                            continue
                        flag_row_matched = True

                    if flag_row_matched:
                        rows_matched.append(row_name)
                        print("\tRow matched: {}".format(row_name))

            statement_type_predict = None

            # candidate: <ranges from NUM to NUM>
            # TODO Handle when unit also is mentioned as part of NUM
            if len(column_matched_tokens_dict) > 0:
                for token_i in range(len(self.statements[stmnt_i].tokens)-4):
                    if (self.statements[stmnt_i].tokens[token_i].lemma == "range") and \
                            (self.statements[stmnt_i].tokens[token_i+1].lemma == "from") and \
                            (self.statements[stmnt_i].tokens[token_i+2].part_of_speech_coarse == "NUM") and \
                            (self.statements[stmnt_i].tokens[token_i+3].lemma == "to") and \
                            (self.statements[stmnt_i].tokens[token_i+4].part_of_speech_coarse == "NUM"):
                        _, min_range_value = is_number(self.statements[stmnt_i].tokens[token_i+2].text)
                        _, max_range_value = is_number(self.statements[stmnt_i].tokens[token_i+4].text)
                        if verbose:
                            print("\tcandidate: <ranges from NUM to NUM> :: token range: ({}, {})".format(token_i, token_i+5))
                        col_index = list(column_matched_tokens_dict.keys())[0]
                        if not is_numeric_dtype(self.df.iloc[:, col_index]):
                            continue
                        token_index_end_column = column_matched_tokens_dict[col_index][-1][1]
                        if token_index_end_column <= token_i:
                            min_column_value = np.nanmin(self.df.iloc[:, col_index])
                            max_column_value = np.nanmax(self.df.iloc[:, col_index])
                            if (min_range_value == min_column_value) and (max_range_value == max_column_value):
                                statement_type_predict = "entailed"
                                if verbose:
                                    print("\t\tcandidate verified")
                            else:
                                statement_type_predict = "refuted"
                                if verbose:
                                    print("\t\tcandidate failed to verify: column value range: ({}, {}) :: statement value range: ({}, {})".format(
                                        min_column_value, max_column_value, min_range_value, max_range_value))

            # candidate: superlative
            m = re.search(r'(\bhighest\b|\bgreatest\b|\blargest\b|\blowest\b)', self.statements[stmnt_i].text,
                          flags=re.I)
            if m and verbose:
                print("\tSuperlative: {}".format(m.group(0)))

            if m and statement_type_predict is None:
                # Identify the numeric column and the numeric superlative value
                numeric_column_index_arr = []
                non_numeric_column_index_arr = []

                for col_index in sorted(column_matched_tokens_dict.keys()):
                    if is_numeric_dtype(self.df.iloc[:, col_index]):
                        numeric_column_index_arr.append(col_index)
                    else:
                        non_numeric_column_index_arr.append(col_index)

                # extract the numeric value associated with superlative candidate in the statement
                # TODO Currently only the final numeric value is considered without any analysis whether it can be associated with superlative candidate or not.
                numeric_value = None

                for token_index_stmnt in range(len(self.statements[stmnt_i].tokens)):
                    cur_token = self.statements[stmnt_i].tokens[token_index_stmnt]
                    # consider only if its not part of the statement tokens matching the column, row
                    if cur_token.part_of_speech_coarse != "NUM":
                        continue

                    flag_stmnt_token_belongs_to_column = False
                    for col_index in column_matched_tokens_dict:
                        for token_index_start_column, token_index_end_column in column_matched_tokens_dict[col_index]:
                            if token_index_stmnt in range(token_index_start_column, token_index_end_column):
                                flag_stmnt_token_belongs_to_column = True
                                break
                        if flag_stmnt_token_belongs_to_column:
                            break

                    if flag_stmnt_token_belongs_to_column:
                        continue

                    flag_stmnt_token_belongs_to_row = False
                    for row_i in range(len(rows_matched)):
                        if token_index_stmnt in range(rows_matched[row_i][1], rows_matched[row_i][2]):
                            flag_stmnt_token_belongs_to_row = True
                            break

                    if flag_stmnt_token_belongs_to_row:
                        continue

                    _, numeric_value = is_number(cur_token.text)

                min_column_value = None
                max_column_value = None
                min_column_value_df_row_index = None
                max_column_value_df_row_index = None

                """
                Cases:
                    (a) TODO Both superlative numeric value and corresponding row mentioned in the statement.
                    (b) Only superlative numeric value mentioned i.e. corresponding row not mentioned.
                    (c) Only superlative row mentioned i.e. corresponding nummeric value not mentioned.
                """
                if numeric_value is not None:
                    if len(numeric_column_index_arr) == 1:
                        col_index = numeric_column_index_arr[0]
                        min_column_value = np.nanmin(self.df.iloc[:, col_index])
                        max_column_value = np.nanmax(self.df.iloc[:, col_index])
                    elif len(numeric_column_index_arr) == 0 and len(non_numeric_column_index_arr) == 1:
                        # case: Due to presence of non-numeric character(s), pandas identified the column with data type: object
                        #       Context: Mention of superlative word for the column indicates that column being numeric.
                        #       Hence identifying the numeric portion from the column cells.
                        col_index = non_numeric_column_index_arr[0]
                        min_column_value, min_column_value_df_row_index, max_column_value, max_column_value_df_row_index = \
                            self.extract_numeric_range_of_non_numeric_column(col_index=col_index)

                    if min_column_value is not None:
                        if m.group(0).lower() == "lowest":
                            if min_column_value == numeric_value:
                                statement_type_predict = "entailed"
                            else:
                                statement_type_predict = "refuted"
                        else:
                            if max_column_value == numeric_value:
                                statement_type_predict = "entailed"
                            else:
                                statement_type_predict = "refuted"

                elif len(rows_matched) == 1:
                    # case: numeric value of superlative not mentioned in the statement
                    # e.g. 20509.xml, Table 1
                    #       Statement: Reagent NaCl salt has highest Concentration .
                    #           Column: Reagent :: row: NaCl
                    #           Column: Concentration :: numeric column

                    col_index = None
                    is_col_numeric = None

                    n_candidate_cols = 0
                    n_candidate_cols += len([x for x in numeric_column_index_arr if x > 0])
                    n_candidate_cols += len([x for x in non_numeric_column_index_arr if x > 0])

                    if n_candidate_cols == 1:
                        for c_index in numeric_column_index_arr:
                            if c_index > 0:
                                col_index = c_index
                                is_col_numeric = True
                                break

                        for c_index in non_numeric_column_index_arr:
                            if c_index > 0:
                                col_index = c_index
                                is_col_numeric = False
                                break

                    if col_index is not None:
                        if is_col_numeric:
                            min_column_value_df_row_index = self.df.iloc[:, col_index].idxmin()
                            max_column_value_df_row_index = self.df.iloc[:, col_index].idxmax()
                        else:
                            min_column_value, min_column_value_df_row_index, max_column_value, max_column_value_df_row_index = \
                                self.extract_numeric_range_of_non_numeric_column(col_index=col_index)

                        if min_column_value_df_row_index is not None :
                            # considering only 0'th column because row names are matched for that only as of now
                            row_index = rows_matched[0][0]

                            if m.group(0).lower() == "lowest":
                                if row_index-self.table_data_start_row_index == min_column_value_df_row_index:
                                    statement_type_predict = "entailed"
                                else:
                                    statement_type_predict = "refuted"
                            else:
                                if row_index-self.table_data_start_row_index == max_column_value_df_row_index:
                                    statement_type_predict = "entailed"
                                else:
                                    statement_type_predict = "refuted"

            # candidate: identical, uniqueness, varies of column
            flag_candidate_identical = False
            flag_candidate_unique = False
            flag_candidate_vary = False

            if len(rows_matched) == 0:
                for token_index_stmnt in range(len(self.statements[stmnt_i].tokens)):
                    cur_token = self.statements[stmnt_i].tokens[token_index_stmnt]
                    if cur_token.lemma.lower() in ["different", "unique"]:
                        flag_candidate_unique = True
                        if verbose:
                            print("\tunique: {}".format(cur_token.text))
                    elif cur_token.lemma.lower() in ["vary", "variation"]:
                        flag_candidate_vary = True
                        if verbose:
                            print("\tvary: {}".format(cur_token.text))
                    elif cur_token.lemma.lower() in ["identical", "same"]:
                        flag_candidate_identical = True

            if flag_candidate_unique and statement_type_predict is None:
                if len(column_matched_tokens_dict) == 1:
                    col_index = list(column_matched_tokens_dict.keys())[0]

                    if len(self.df.iloc[:, col_index].unique()) == len(self.df):
                        statement_type_predict = "entailed"
                    else:
                        statement_type_predict = "refuted"

            if flag_candidate_vary and statement_type_predict is None:
                if len(column_matched_tokens_dict) == 1:
                    col_index = list(column_matched_tokens_dict.keys())[0]
                    if len(self.df.iloc[:, col_index].unique()) > 1:
                        statement_type_predict = "entailed"
                    else:
                        statement_type_predict = "refuted"

            if False:
                m = re.search(r'(\bunique\b)', self.statements[stmnt_i].text, flags=re.I)

                if m and verbose:
                    print("\tUniqueness: {}".format(m.group(0)))

                if m and statement_type_predict is None:
                    if len(column_matched_tokens_dict) == 1:
                        col_index = list(column_matched_tokens_dict.keys())[0]

                        if len(self.df.iloc[:, col_index].unique()) == len(self.df):
                            statement_type_predict = "entailed"
                        else:
                            statement_type_predict = "refuted"

            # candidate: count rows for column
            if len(column_matched_tokens_dict) == 1 and statement_type_predict is None:
                col_index = list(column_matched_tokens_dict.keys())[0]
                n_rows_col = len(self.df.iloc[:, col_index].dropna())
                for token_index_stmnt in range(len(self.statements[stmnt_i].tokens)):
                    cur_token = self.statements[stmnt_i].tokens[token_index_stmnt]
                    if cur_token.part_of_speech_coarse == "NUM" and cur_token.dependency_tag == "nummod":
                        # consider only if its not part of the statement tokens matching the column, row
                        col_index = list(column_matched_tokens_dict.keys())[0]
                        flag_stmnt_token_belongs_to_column = False

                        for token_index_start_column, token_index_end_column in column_matched_tokens_dict[col_index]:
                            if token_index_stmnt in range(token_index_start_column, token_index_end_column):
                                flag_stmnt_token_belongs_to_column = True
                                break
                        if flag_stmnt_token_belongs_to_column:
                            continue

                        flag_stmnt_token_belongs_to_row = False
                        for row_i in range(len(rows_matched)):
                            if token_index_stmnt in range(rows_matched[row_i][1], rows_matched[row_i][2]):
                                flag_stmnt_token_belongs_to_row = True
                                break

                        if flag_stmnt_token_belongs_to_row:
                            continue

                        flag_numeric, statement_token_value = is_number(cur_token.text)
                        flag_match = None
                        if flag_numeric:
                            # Basic check to skip considering float aa count
                            # TODO Need to identify text/dependency tree pattern to consider as candidate for count rows
                            if int(statement_token_value) == statement_token_value:
                                flag_match = statement_token_value == n_rows_col
                        else:
                            # convert word to numeric value
                            try:
                                statement_token_value_numeric = w2n.word_to_num(statement_token_value)
                                flag_match = n_rows_col == statement_token_value_numeric
                            except:
                                continue

                        if flag_match is True:
                            statement_type_predict = "entailed"
                        elif flag_match is False:
                            statement_type_predict = "refuted"

                        if verbose:
                            print("\tNumber of rows for column: {} :: count mentioned in statement: {} :: match: {}".format(n_rows_col, statement_token_value, flag_match))

            # candidate: Whether two rows are same/different for a column
            m = re.search(r'(\bsame\b|\bdifferent\b)', self.statements[stmnt_i].text, flags=re.I)

            if m and len(rows_matched) == 2 and statement_type_predict is None:
                col_index = None
                candidate_cols = [x for x in column_matched_tokens_dict if x > 0]

                if len(candidate_cols) == 1:
                    col_index = candidate_cols[0]

                if col_index is not None:
                    row_index_0 = rows_matched[0][0]
                    row_index_1 = rows_matched[1][0]
                    cell_value_0 = self.df.iloc[row_index_0 - self.table_data_start_row_index, col_index]
                    cell_value_1 = self.df.iloc[row_index_1 - self.table_data_start_row_index, col_index]

                    if m.group(0).lower() == "same":
                        if cell_value_0 == cell_value_1:
                            statement_type_predict = "entailed"
                        else:
                            statement_type_predict = "refuted"
                    else:
                        if cell_value_0 != cell_value_1:
                            statement_type_predict = "entailed"
                        else:
                            statement_type_predict = "refuted"

            # candidate: cell value
            if len(rows_matched) == 1 and statement_type_predict is None:
                numeric_column_index_arr = []
                non_numeric_column_index_arr = []

                for c_index in sorted(column_matched_tokens_dict.keys()):
                    if is_numeric_dtype(self.df.iloc[:, c_index]):
                        numeric_column_index_arr.append(c_index)
                    else:
                        non_numeric_column_index_arr.append(c_index)

                col_index = None
                is_col_numeric = None

                n_candidate_cols = 0
                n_candidate_cols += len([x for x in numeric_column_index_arr if x > 0])
                n_candidate_cols += len([x for x in non_numeric_column_index_arr if x > 0])

                if n_candidate_cols == 1:
                    for c_index in numeric_column_index_arr:
                        if c_index > 0:
                            col_index = c_index
                            is_col_numeric = True
                            break

                    for c_index in non_numeric_column_index_arr:
                        if c_index > 0:
                            col_index = c_index
                            is_col_numeric = False
                            break

                if col_index is not None:
                    row_index = rows_matched[0][0]
                    token_index_start_row = rows_matched[0][1]
                    token_index_end_row = rows_matched[0][2]

                    if row_index in self.cell_info_dict and col_index in self.cell_info_dict[row_index]:
                        match_length = len(self.cell_info_dict[row_index][col_index])

                        flag_cell_matched = None
                        for token_index_stmnt in range(len(self.statements[stmnt_i].tokens) - match_length + 1):
                            token_i = 0
                            flag_cell_matched = True
                            while token_i < match_length:
                                if (self.statements[stmnt_i].tokens[token_index_stmnt + token_i].text.lower() !=
                                        self.cell_info_dict[row_index][col_index][token_i].text.lower()) and \
                                        (self.statements[stmnt_i].tokens[token_index_stmnt + token_i].lemma.lower() !=
                                             self.cell_info_dict[row_index][col_index][token_i].lemma.lower()):
                                    flag_cell_matched = False
                                    break
                                token_i += 1

                            if flag_cell_matched:
                                if verbose:
                                    cell_text = " ".join([self.cell_info_dict[row_index][col_index][i].text for i in range(len(self.cell_info_dict[row_index][col_index]))])
                                    print("\tCell matched: {}".format(cell_text))

                                break

                        if flag_cell_matched:
                            statement_type_predict = "entailed"
                        elif not flag_cell_matched:
                            statement_type_predict = "refuted"

            statement_id_predict_type_map[self.statements[stmnt_i].id] = statement_type_predict

            if verbose:
                print("\tPredicted type: {}".format(statement_type_predict))

            if True:
                print("Statement: id: {} :: type(ground truth): {} :: type(predicted): {} :: text: {}".format(
                    self.statements[stmnt_i].id, self.statements[stmnt_i].type, statement_type_predict, self.statements[stmnt_i].text))

        return statement_id_predict_type_map

    def build_table_element(self, statement_id_predict_type_map):
        """Build XML table element with predictions.
            Populate only the fields which are absolutely required for evaluation.

            Returns
            -------
            table xml element
        """
        table_elem = ET.Element("table")
        table_elem.set("id", self.table_id)
        statements_elem = ET.Element("statements")
        table_elem.append(statements_elem)

        for stmnt_i in range(len(self.statements)):
            statement_elem = ET.SubElement(statements_elem, "statement")
            statement_elem.set("id", self.statements[stmnt_i].id)
            # statement_elem.set("text", self.statements[stmnt_i].text)
            if self.statements[stmnt_i].id in statement_id_predict_type_map:
                statement_type_predict = statement_id_predict_type_map[self.statements[stmnt_i].id]
            else:
                statement_type_predict = None

            if statement_type_predict is None:
                statement_type_predict = "unknown"

            statement_elem.set("type", statement_type_predict)

        return table_elem

    def extract_numeric_range_of_non_numeric_column(self, col_index):
        min_column_value = None
        max_column_value = None
        min_column_value_df_row_index = None
        max_column_value_df_row_index = None

        for df_row_index, column_value in enumerate(self.df.iloc[:, col_index].values):
            if column_value is None:
                continue
            if isinstance(column_value, float) and np.isnan(column_value):
                continue
            # extract the numeric portion (if available)
            # TODO Cost value: $ <number>
            flag_numeric = False
            if isinstance(column_value, str):
                column_value_mod = re.sub(r'[,\s]', r'', column_value)
                m_col_value = re.match(r'\d+', column_value_mod)
                if m_col_value is None:
                    continue
                flag_numeric, numeric_column_value = is_number(m_col_value.group(0))
            elif isinstance(column_value, int) or isinstance(column_value, float):
                flag_numeric = True
                numeric_column_value = column_value

            if flag_numeric:
                if min_column_value is None:
                    min_column_value = numeric_column_value
                    max_column_value = numeric_column_value
                    min_column_value_df_row_index = df_row_index
                    max_column_value_df_row_index = df_row_index
                else:
                    if numeric_column_value < min_column_value:
                        min_column_value = min_column_value
                        min_column_value_df_row_index = df_row_index

                    if max_column_value < numeric_column_value:
                        max_column_value = numeric_column_value
                        max_column_value_df_row_index = df_row_index

        return min_column_value, min_column_value_df_row_index,  max_column_value, max_column_value_df_row_index
