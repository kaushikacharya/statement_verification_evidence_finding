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
    def __init__(self, text, lemma=None, normalized_text=None, start_char_offset=None, part_of_speech_coarse=None,
                 part_of_speech_fine_grained=None, dependency_tag=None, head_index=None, children_index_arr=None):
        self.text = text
        self.lemma = lemma
        self.normalized_text = normalized_text
        # character offset of the token in the parent document
        self.start_char_offset = start_char_offset
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
        self.empty_col_set = None
        # dataframe to store table.
        #   Table row indices in range (0, self.table_data_start_row_index) represents column headers.
        #   Table row indices in range (self.table_data_start_row_index, self.table_data_end_row_index) represents values.
        self.df = None
        # dict of dict: keys: [row_index][col_index],  values: list of Token
        self.cell_info_dict = dict()
        # dict of dict: keys: [row_index][col_index],  values: set of statement ids for which the cell is relevant
        self.cell_evidence_dict = dict()
        self.statements = []
        self.jaro_similarity_threshold = 0.85

    def parse_xml(self, table_item, flag_cell_span=True, verbose=False):
        """Parse table element of xml.
            Identify the table rows for column names.

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

        if len(self.caption_text) > 0:
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

                self.cell_info_dict[cur_row_id][cur_col_id] = dict()
                # insert empty list which will be populated with list of Token objects
                self.cell_info_dict[cur_row_id][cur_col_id]["tokens"] = []
                if flag_cell_span:
                    # N.B. [col_begin, col_end) represents range i.e. end is excluded
                    #   Whereas [col-start, col-end] is inclusive of end
                    self.cell_info_dict[cur_row_id][cur_col_id]["col_begin"] = int(cell.attrib["col-start"])
                    self.cell_info_dict[cur_row_id][cur_col_id]["col_end"] = int(cell.attrib["col-end"]) + 1
                    self.cell_info_dict[cur_row_id][cur_col_id]["row_begin"] = int(cell.attrib["row-start"])
                    self.cell_info_dict[cur_row_id][cur_col_id]["row_end"] = int(cell.attrib["row-end"]) + 1

                if cur_row_id not in self.cell_evidence_dict:
                    self.cell_evidence_dict[cur_row_id] = dict()
                # insert empty set which will be later populated with statement ids for which the cell is relevant
                self.cell_evidence_dict[cur_row_id][cur_col_id] = set()

                doc_cell = self.nlp_process_obj.construct_doc(text=cell_text)
                for token in doc_cell:
                    token_obj = Token(text=token.text, lemma=token.lemma_, normalized_text=token.norm_,
                                      start_char_offset=token.idx,
                                      part_of_speech_coarse=token.pos_, part_of_speech_fine_grained=token.tag_,
                                      dependency_tag=token.dep_, head_index=token.head.i,
                                      children_index_arr=[child.i for child in token.children])
                    self.cell_info_dict[cur_row_id][cur_col_id]["tokens"].append(token_obj)
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
                self.empty_col_set = empty_col_set
                break
            else:
                # case: Empty columns in-between columns with data filled
                #   Re-process above code segment to identify from which row table data starts
                #   e.g. 20690.xml, Table 1, Table 3
                #   In 2nd iteration these internal empty columns are not processed for the purpose of identifying
                #       row from which table data starts.
                # TODO Also these empty columns could be a signal of range for nested column headers.
                self.empty_col_set = empty_col_set
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
            table_id_tokens = self.table_id.split()
            # m = re.search(r'\d+', self.table_id)
            # assert m, "Table id: {} does not contain numerals".format(self.table_id)
            output_dir = os.path.join(os.path.dirname(__file__), "../output/debug", self.doc_id, table_id_tokens[-1])  # m.group(0)
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
                                      start_char_offset=token.idx,
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

    def process_table(self, flag_approx_string_match=False, verbose=False):
        """Process table for statement verification and evidence finding

            Parameters
            ----------
            flag_approx_string_match : bool
            verbose : bool
        """

        # key: statement id  value: predicted type
        statement_id_predict_type_map = dict()

        if verbose:
            print("\n")

        for stmnt_i in range(len(self.statements)):
            if verbose:
                print("Statement #{} :: id: {} :: type(ground truth): {} :: text: {}".format(
                    stmnt_i, self.statements[stmnt_i].id, self.statements[stmnt_i].type, self.statements[stmnt_i].text))

            statement_type_predict = self.process_statement(stmnt_i=stmnt_i, flag_approx_string_match=flag_approx_string_match, verbose=verbose)

            statement_id_predict_type_map[self.statements[stmnt_i].id] = statement_type_predict

            if True:
                print("\tStatement: id: {} :: type(ground truth): {} :: type(predicted): {} :: text: {}".format(
                    self.statements[stmnt_i].id, self.statements[stmnt_i].type, statement_type_predict, self.statements[stmnt_i].text))

        return statement_id_predict_type_map

    def process_statement(self, stmnt_i, flag_approx_string_match=False, verbose=False):
        """Process statement for statement verification and evidence finding

            Parameters
            ----------
            stmnt_i : int (statement index)
            flag_approx_string_match : bool
            verbose : bool
        """
        statement_type_predict = None

        # Identify columns in statement

        # iterate over the table cell corresponding to the column headers
        flag_col_matched_arr = [None for i in range(len(self.df.columns))]

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
                if self.cell_info_dict[row_index][col_index]["tokens"][-1].part_of_speech_coarse == "PUNCT":
                    begin_optional_text = None
                    if self.cell_info_dict[row_index][col_index]["tokens"][-1].text == ")":
                        begin_optional_text = "("
                    elif self.cell_info_dict[row_index][col_index]["tokens"][-1].text == "]":
                        begin_optional_text = "["

                    if begin_optional_text:
                        token_i = 1
                        while token_i < len(self.cell_info_dict[row_index][col_index]["tokens"]):
                            if self.cell_info_dict[row_index][col_index]["tokens"][token_i].text == begin_optional_text:
                                optional_token_begin = token_i
                                break
                            token_i += 1

                match_token_length = optional_token_begin if optional_token_begin else len(
                    self.cell_info_dict[row_index][col_index]["tokens"])

                col_cell_match_text = ""
                prev_char_offset = 0
                for token_i in range(match_token_length):
                    col_cell_match_text += " " * (
                    self.cell_info_dict[row_index][col_index]["tokens"][token_i].start_char_offset - prev_char_offset)
                    col_cell_match_text += self.cell_info_dict[row_index][col_index]["tokens"][token_i].text
                    prev_char_offset = self.cell_info_dict[row_index][col_index]["tokens"][token_i].start_char_offset + \
                                       len(self.cell_info_dict[row_index][col_index]["tokens"][token_i].text)

                cur_column_matched_tokens_dict = dict()
                for token_index_stmnt in range(len(self.statements[stmnt_i].tokens) - match_token_length + 1):
                    # token_index_match_end_stmnt = token_index_stmnt + match_token_length
                    char_pos_match_stmnt_begin = self.statements[stmnt_i].tokens[token_index_stmnt].start_char_offset
                    char_pos_match_stmnt_end = self.statements[stmnt_i].tokens[
                                                   token_index_stmnt + match_token_length - 1].start_char_offset + \
                                               len(self.statements[stmnt_i].tokens[
                                                       token_index_stmnt + match_token_length - 1].text)
                    stmnt_match_text = self.statements[stmnt_i].text[
                                       char_pos_match_stmnt_begin: char_pos_match_stmnt_end]

                    # N.B. token_i variable could be confusing. It refers to token index in the column name cell and not of the statement tokens.
                    token_i = 0
                    jaro_sim = None
                    if flag_approx_string_match:
                        jaro_sim = jaro_similarity(col_cell_match_text.lower(), stmnt_match_text.lower())

                        if jaro_sim < self.jaro_similarity_threshold:
                            flag_col_cell_matched = False
                        else:
                            flag_col_cell_matched = True

                        # update token_i
                        token_i = match_token_length
                    else:
                        flag_col_cell_matched = True
                        while token_i < match_token_length:
                            if (self.statements[stmnt_i].tokens[token_index_stmnt + token_i].text.lower() !=
                                    self.cell_info_dict[row_index][col_index]["tokens"][token_i].text.lower()) and \
                                    (self.statements[stmnt_i].tokens[token_index_stmnt + token_i].lemma.lower() !=
                                         self.cell_info_dict[row_index][col_index]["tokens"][token_i].lemma.lower()):
                                flag_col_cell_matched = False
                                break
                            token_i += 1

                    if flag_col_cell_matched:
                        if optional_token_begin:
                            # The optional part of the column were not matched above with statement tokens.
                            #  Here we would continue matching that part as long as it matches.
                            # case (a): All the column tokens get matched.
                            # case (b): Matches a certain portion of optional portion.
                            #       e.g. 20758.xml, Table 1
                            #       Non-ascii tokens are present in column name but absent in statement.
                            #       This could be because these non-ascii tokens for columns were scraped from web
                            #          but statements were typed by volunteers.
                            flag_col_cell_full_matched = False
                            token_j = token_i

                            while (token_i < min(len(self.cell_info_dict[row_index][col_index]["tokens"]),
                                                 len(self.statements[stmnt_i].tokens) - token_index_stmnt)):
                                # Second argument of min ensures that we don't go beyond the statement tokens
                                if (self.statements[stmnt_i].tokens[token_index_stmnt + token_i].text.lower() !=
                                        self.cell_info_dict[row_index][col_index]["tokens"][token_i].text.lower()) and \
                                        (self.statements[stmnt_i].tokens[token_index_stmnt + token_i].lemma.lower() !=
                                             self.cell_info_dict[row_index][col_index]["tokens"][
                                                 token_i].lemma.lower()):
                                    break
                                else:
                                    # update column cell matched text
                                    col_cell_match_text += " " * (self.cell_info_dict[row_index][col_index]["tokens"][
                                                                      token_i].start_char_offset - prev_char_offset)
                                    col_cell_match_text += self.cell_info_dict[row_index][col_index]["tokens"][
                                        token_i].text
                                    prev_char_offset = self.cell_info_dict[row_index][col_index]["tokens"][
                                                           token_i].start_char_offset + \
                                                       len(self.cell_info_dict[row_index][col_index]["tokens"][
                                                               token_i].text)

                                token_i += 1

                            if token_i == len(self.cell_info_dict[row_index][col_index]["tokens"]):
                                flag_col_cell_full_matched = True

                            if token_i > token_j:
                                # update statement matched text
                                char_pos_match_stmnt_end = self.statements[stmnt_i].tokens[
                                                               token_index_stmnt + token_i - 1].start_char_offset + \
                                                           len(self.statements[stmnt_i].tokens[
                                                                   token_index_stmnt + token_i - 1].text)
                                stmnt_match_text = self.statements[stmnt_i].text[
                                                   char_pos_match_stmnt_begin: char_pos_match_stmnt_end]
                                if flag_approx_string_match:
                                    jaro_sim = jaro_similarity(col_cell_match_text.lower(), stmnt_match_text.lower())
                        else:
                            flag_col_cell_full_matched = True

                        if 'token_index_range_statement' not in cur_column_matched_tokens_dict:
                            cur_column_matched_tokens_dict['token_index_range_statement'] = []
                            cur_column_matched_tokens_dict['score'] = []

                        # case a: Current matched tokens don't overlap with previous matched tokens (if available)
                        #           Add the current matched tokens
                        # case b: Overlap between current matched tokens with previous matched tokens
                        #           If current one has better similarity score then replace the previous one
                        flag_append_cur_matched_tokens = True
                        flag_update_prev_matched_tokens = False
                        if len(cur_column_matched_tokens_dict[
                                   'token_index_range_statement']) > 0 and jaro_sim is not None:
                            if token_index_stmnt < cur_column_matched_tokens_dict['token_index_range_statement'][-1][1]:
                                flag_append_cur_matched_tokens = False
                                if jaro_sim > cur_column_matched_tokens_dict['score'][-1]:
                                    cur_column_matched_tokens_dict['token_index_range_statement'][-1] = (
                                    token_index_stmnt, token_index_stmnt + token_i)
                                    cur_column_matched_tokens_dict['score'][-1] = jaro_sim

                        if flag_append_cur_matched_tokens:
                            cur_column_matched_tokens_dict['token_index_range_statement'].append(
                                (token_index_stmnt, token_index_stmnt + token_i))
                            cur_column_matched_tokens_dict['score'].append(1.0 if jaro_sim is None else jaro_sim)

                        if verbose:
                            col_info = (col_index, token_index_stmnt, token_index_stmnt + token_i)

                            col_cell_text = ""
                            prev_char_offset = 0
                            for x in self.cell_info_dict[row_index][col_index]["tokens"]:
                                col_cell_text += " " * (x.start_char_offset - prev_char_offset)
                                col_cell_text += x.text
                                prev_char_offset = x.start_char_offset + len(x.text)

                            # col_cell_text = " ".join([x.text for x in self.cell_info_dict[row_index][col_index]["tokens"]])
                            print_text = "\tColumn cell matched: {} :: col info: {}".format(col_cell_text, col_info)
                            if jaro_sim is not None:
                                print_text += " :: similarity score: {:.3f}".format(jaro_sim)
                            print_text += " :: statement text matched: {}".format(stmnt_match_text)
                            print(print_text)
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
                            column_matched_tokens_dict[col_index] = dict()
                        if row_index not in column_matched_tokens_dict[col_index]:
                            column_matched_tokens_dict[col_index][row_index] = {"token_index_range_statement": [],
                                                                                "score": []}

                        column_matched_tokens_dict[col_index][row_index]["token_index_range_statement"].extend(
                            cur_column_matched_tokens_dict["token_index_range_statement"])
                        column_matched_tokens_dict[col_index][row_index]["score"].extend(
                            cur_column_matched_tokens_dict["score"])

                        # Assign matched tokens info to other column(s) if the current column cell ranges over multiple consecutive cells.
                        #   This is observed in multi-index column covering multiple sub-columns.
                        #   Also note that cell_info_dict has been populated only for col-start, row-start cell.
                        if "col_begin" in self.cell_info_dict[row_index][col_index] and "col_end" in \
                                self.cell_info_dict[row_index][col_index]:
                            for col_idx in range(self.cell_info_dict[row_index][col_index]["col_begin"] + 1,
                                                 self.cell_info_dict[row_index][col_index]["col_end"]):
                                if col_idx not in column_matched_tokens_dict:
                                    column_matched_tokens_dict[col_idx] = dict()
                                column_matched_tokens_dict[col_idx][row_index] = \
                                    column_matched_tokens_dict[col_index][row_index]
                else:
                    flag_col_matched_arr[col_index] = False
                    # remove if it was added in any of the previous column header row
                    if col_index in column_matched_tokens_dict:
                        column_matched_tokens_dict.pop(col_index)

        # sort in terms of descending score
        for col_index in column_matched_tokens_dict:
            for row_index in column_matched_tokens_dict[col_index]:
                if len(column_matched_tokens_dict[col_index][row_index]["score"]) < 2:
                    continue
                sorted_index_arr = sorted(range(len(column_matched_tokens_dict[col_index][row_index]["score"])),
                                          key=lambda i: (
                                          -1 * column_matched_tokens_dict[col_index][row_index]["score"][i],
                                          column_matched_tokens_dict[col_index][row_index][
                                              "token_index_range_statement"][i]))
                column_matched_tokens_dict[col_index][row_index]["score"] = [
                    column_matched_tokens_dict[col_index][row_index]["score"][i] for i in sorted_index_arr]
                column_matched_tokens_dict[col_index][row_index]["token_index_range_statement"] = [
                    column_matched_tokens_dict[col_index][row_index]["token_index_range_statement"][i] for i in
                    sorted_index_arr]

        """
        if len(column_matched_tokens_dict) > 1:
            # remove column(s) which are subset of another column in terms of statement tokens
            col_index_arr = [k for k, v in sorted(column_matched_tokens_dict.items(), key=lambda x: x[1])]
            prev_col_index = col_index_arr[0]
            for cur_col_index in col_index_arr[1:]:
                flag_prev_col_contains_cur_col = False
                flag_cur_col_contains_prev_col = False

                if (len(column_matched_tokens_dict[cur_col_index]) == 1) and (len(column_matched_tokens_dict[prev_col_index]) == 1):
                    # Currently considering subset only for non-disjoint text span
                    if (column_matched_tokens_dict[cur_col_index][0][0] < column_matched_tokens_dict[prev_col_index][0][1]) and \
                            (column_matched_tokens_dict[cur_col_index][0][1] <= column_matched_tokens_dict[prev_col_index][0][1]):
                        flag_prev_col_contains_cur_col = True
                    elif (column_matched_tokens_dict[cur_col_index][0][0] == column_matched_tokens_dict[prev_col_index][0][0]) and \
                            (column_matched_tokens_dict[cur_col_index][0][1] > column_matched_tokens_dict[prev_col_index][0][1]):
                        # ?? 2nd condition might be redundant since the sort over tuples should have taken care of this
                        flag_cur_col_contains_prev_col = True

                    if flag_prev_col_contains_cur_col:
                        flag_col_matched_arr[cur_col_index] = False
                        column_matched_tokens_dict.pop(cur_col_index)
                    elif flag_cur_col_contains_prev_col:
                        flag_col_matched_arr[prev_col_index] = False
                        column_matched_tokens_dict.pop(prev_col_index)
                        prev_col_index = cur_col_index
                    else:
                        prev_col_index = cur_col_index
        """

        selected_column_index_arr = self.select_matched_columns(stmnt_i=stmnt_i,
                                                                column_matched_tokens_dict=column_matched_tokens_dict)

        if verbose:
            """
            col_matched_arr = [self.df.columns[i] for i in range(len(flag_col_matched_arr)) if flag_col_matched_arr[i]]
            if len(col_matched_arr) > 0:
                print("\tColumns matched: {}".format(col_matched_arr))
            """
            col_matched_arr = [self.df.columns[i] for i in selected_column_index_arr]
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
            for token_index_stmnt in range(len(self.statements[stmnt_i].tokens) - len(
                    self.cell_info_dict[row_index][col_index]["tokens"]) + 1):
                token_i = 0
                flag_row_matched = True
                while token_i < len(self.cell_info_dict[row_index][col_index]["tokens"]):
                    if self.statements[stmnt_i].tokens[token_index_stmnt + token_i].text.lower() != \
                            self.cell_info_dict[row_index][col_index]["tokens"][token_i].text.lower():
                        flag_row_matched = False
                        break
                    token_i += 1

                if flag_row_matched:
                    row_info = (row_index, token_index_stmnt, token_index_stmnt + token_i)
                    rows_matched.append(row_info)
                    row_name = self.df.iloc[row_index - self.table_data_start_row_index, col_index]
                    # row_name = self.df[self.df.columns[0]][row_index-self.table_data_start_row_index]
                    if verbose:
                        print("\tRow matched: name: {} :: row info: {}".format(row_name, row_info))

        # candidate: <numeric value of column mentioned(single data row)>
        #       case (a): <column name> value is <numeric>  e.g. 20758.xml, Table 1, Statement id=5
        #       case (b): <numeric> is the value of/for <column name>  e.g. 20758.xml, Table 1, Statement id=2
        if (len(selected_column_index_arr) == 1) and (
                self.table_data_end_row_index - self.table_data_start_row_index == 1):
            col_index = selected_column_index_arr[0]
            if is_numeric_dtype(self.df.iloc[:, col_index]):
                numeric_value = None
                for token_i in range(len(self.statements[stmnt_i].tokens)):
                    if self.statements[stmnt_i].tokens[token_i].part_of_speech_coarse != "NUM":
                        continue
                    if self.statements[stmnt_i].tokens[token_i].dependency_tag in ["attr", "nsubj"]:
                        head_token_index = self.statements[stmnt_i].tokens[token_i].head_index
                        if self.statements[stmnt_i].tokens[head_token_index].part_of_speech_coarse == "AUX":
                            other_child_dep_tag = "nsubj" if self.statements[stmnt_i].tokens[
                                                                 token_i].dependency_tag == "attr" else "attr"
                            for child_token_index in self.statements[stmnt_i].tokens[
                                head_token_index].children_index_arr:
                                if child_token_index == token_i:
                                    continue
                                # "-" in "p -value" should have been a separate token. But to handle this failure, considering "-value" as token also.
                                if re.match(r"[-]?value", self.statements[stmnt_i].tokens[child_token_index].lemma) and \
                                                self.statements[stmnt_i].tokens[
                                                    child_token_index].dependency_tag == other_child_dep_tag:
                                    _, numeric_value = is_number(self.statements[stmnt_i].tokens[token_i].text)

                if numeric_value is not None:
                    if verbose:
                        print("\tcandidate: numeric value of column mentioned(single data row)")

                    if numeric_value == self.df.iloc[0, col_index]:
                        statement_type_predict = "entailed"
                        if verbose:
                            print("\t\tcandidate verified")
                    else:
                        statement_type_predict = "refuted"
                        if verbose:
                            print("\t\tcandidate failed to verify")

                    # assign evidence
                    for row_idx in range(self.table_data_end_row_index):
                        if row_idx in self.cell_evidence_dict and col_index in self.cell_evidence_dict[row_idx]:
                            self.cell_evidence_dict[row_idx][col_index].add(self.statements[stmnt_i].id)

        # candidate: <ranges from NUM to NUM>
        # TODO Handle when unit also is mentioned as part of NUM
        if statement_type_predict is None and len(selected_column_index_arr) > 0:
            for token_i in range(len(self.statements[stmnt_i].tokens) - 4):
                if (self.statements[stmnt_i].tokens[token_i].lemma == "range") and \
                        (self.statements[stmnt_i].tokens[token_i + 1].lemma == "from") and \
                        (self.statements[stmnt_i].tokens[token_i + 2].part_of_speech_coarse == "NUM") and \
                        (self.statements[stmnt_i].tokens[token_i + 3].lemma == "to") and \
                        (self.statements[stmnt_i].tokens[token_i + 4].part_of_speech_coarse == "NUM"):
                    _, min_range_value = is_number(self.statements[stmnt_i].tokens[token_i + 2].text)
                    _, max_range_value = is_number(self.statements[stmnt_i].tokens[token_i + 4].text)
                    if verbose:
                        print("\tcandidate: <ranges from NUM to NUM> :: token range: ({}, {})".format(token_i, token_i + 5))
                    col_index = selected_column_index_arr[0]
                    if not is_numeric_dtype(self.df.iloc[:, col_index]):
                        continue
                    token_index_end_column = -1
                    for row_index in column_matched_tokens_dict[col_index]:
                        token_index_end_column = max(
                            column_matched_tokens_dict[col_index][row_index]["token_index_range_statement"][0][1],
                            token_index_end_column)

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
                                print(
                                    "\t\tcandidate failed to verify: column value range: ({}, {}) :: statement value range: ({}, {})".format(
                                        min_column_value, max_column_value, min_range_value, max_range_value))

                        # assign evidence
                        for row_index in range(self.table_data_end_row_index):
                            if row_index in self.cell_evidence_dict and col_index in self.cell_evidence_dict[row_index]:
                                self.cell_evidence_dict[row_index][col_index].add(self.statements[stmnt_i].id)

        # candidate: superlative
        m = re.search(r'(\bhighest\b|\bgreatest\b|\blargest\b|\blowest\b)', self.statements[stmnt_i].text,
                      flags=re.I)
        if m and verbose:
            print("\tSuperlative: {}".format(m.group(0)))

        if m and statement_type_predict is None:
            # Identify the numeric column and the numeric superlative value
            numeric_column_index_arr = []
            non_numeric_column_index_arr = []

            for col_index in sorted(selected_column_index_arr):
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
                for col_index in selected_column_index_arr:
                    for row_index in column_matched_tokens_dict[col_index]:
                        for token_index_start_column, token_index_end_column in \
                        column_matched_tokens_dict[col_index][row_index]["token_index_range_statement"]:
                            if token_index_stmnt in range(token_index_start_column, token_index_end_column):
                                flag_stmnt_token_belongs_to_column = True
                                break
                        if flag_stmnt_token_belongs_to_column:
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
                (c) Only superlative row mentioned i.e. corresponding numeric value not mentioned.
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
                    # Identified min/max column value
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

                    # assign evidence
                    for row_idx in range(self.table_data_end_row_index):
                        if row_idx in self.cell_evidence_dict and col_index in self.cell_evidence_dict[row_idx]:
                            self.cell_evidence_dict[row_idx][col_index].add(self.statements[stmnt_i].id)

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

                    if min_column_value_df_row_index is not None:
                        # considering only 0'th column because row names are matched for that only as of now
                        row_index = rows_matched[0][0]

                        if m.group(0).lower() == "lowest":
                            if row_index - self.table_data_start_row_index == min_column_value_df_row_index:
                                statement_type_predict = "entailed"
                            else:
                                statement_type_predict = "refuted"
                        else:
                            if row_index - self.table_data_start_row_index == max_column_value_df_row_index:
                                statement_type_predict = "entailed"
                            else:
                                statement_type_predict = "refuted"

                        # assign evidence to the row matched
                        if row_index in self.cell_evidence_dict and 0 in self.cell_evidence_dict[row_index]:
                            self.cell_evidence_dict[row_index][0].add(self.statements[stmnt_i].id)

                        # assign evidence for the column header mentioning the row name
                        for row_idx in range(self.table_data_start_row_index):
                            if row_idx in self.cell_evidence_dict and 0 in self.cell_evidence_dict[row_idx]:
                                self.cell_evidence_dict[row_idx][0].add(self.statements[stmnt_i].id)

                        # assign evidence for the column for which superlative candidate was matched
                        for row_idx in range(self.table_data_end_row_index):
                            if row_idx in self.cell_evidence_dict and col_index in self.cell_evidence_dict[row_idx]:
                                self.cell_evidence_dict[row_idx][col_index].add(self.statements[stmnt_i].id)

        # candidate: identical, uniqueness, varies of column
        # sub-candidate: count of different values mentioned
        flag_candidate_identical = False
        flag_candidate_unique = False
        flag_candidate_vary = False
        n_unique_row = None

        if len(rows_matched) == 0:
            prev_token = None
            for token_index_stmnt in range(len(self.statements[stmnt_i].tokens)):
                cur_token = self.statements[stmnt_i].tokens[token_index_stmnt]
                if cur_token.lemma.lower() in ["different", "unique"]:
                    flag_candidate_unique = True
                    if prev_token and prev_token.part_of_speech_coarse == "NUM":
                        # e.g. 20506.xml, Table 2
                        #   Statement: There are 2 different types of measures.
                        #   Column: Measure
                        flag_numeric, statement_prev_token_value = is_number(prev_token.text)
                        if flag_numeric:
                            n_unique_row = statement_prev_token_value
                        else:
                            # convert word to numeric value
                            try:
                                n_unique_row = w2n.word_to_num(statement_prev_token_value)
                            except:
                                pass

                    if verbose:
                        print("\tunique: {} {}".format(n_unique_row, cur_token.text))
                elif cur_token.lemma.lower() in ["vary", "variation"]:
                    flag_candidate_vary = True
                    if verbose:
                        print("\tvary: {}".format(cur_token.text))
                elif cur_token.lemma.lower() in ["identical", "same"]:
                    flag_candidate_identical = True
                    if verbose:
                        print("\tidentical: {}".format(cur_token.text))

                # update previous token before next iteration
                prev_token = cur_token

        if flag_candidate_unique and statement_type_predict is None:
            if len(selected_column_index_arr) == 1:
                col_index = selected_column_index_arr[0]
                # expected unique count is based on whether number of different values of column is mentioned in the statement or not
                n_expected_unique_row = n_unique_row if n_unique_row is not None else len(self.df)

                if len(self.df.iloc[:, col_index].dropna().unique()) == n_expected_unique_row:
                    statement_type_predict = "entailed"
                else:
                    statement_type_predict = "refuted"

                # assign evidence
                for row_idx in range(self.table_data_end_row_index):
                    if row_idx in self.cell_evidence_dict and col_index in self.cell_evidence_dict[row_idx]:
                        self.cell_evidence_dict[row_idx][col_index].add(self.statements[stmnt_i].id)

        if flag_candidate_vary and statement_type_predict is None:
            if len(selected_column_index_arr) == 1:
                col_index = selected_column_index_arr[0]
                if len(self.df.iloc[:, col_index].dropna().unique()) > 1:
                    statement_type_predict = "entailed"
                else:
                    statement_type_predict = "refuted"

                # assign evidence
                for row_idx in range(self.table_data_end_row_index):
                    if row_idx in self.cell_evidence_dict and col_index in self.cell_evidence_dict[row_idx]:
                        self.cell_evidence_dict[row_idx][col_index].add(self.statements[stmnt_i].id)

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

        # candidate: count columns in the table
        #   TODO Verify if column names also mentioned
        if statement_type_predict is None:
            for token_index_stmnt in range(len(self.statements[stmnt_i].tokens)):
                cur_token = self.statements[stmnt_i].tokens[token_index_stmnt]
                if cur_token.part_of_speech_coarse == "NUM" and cur_token.dependency_tag == "nummod":
                    head_token = self.statements[stmnt_i].tokens[cur_token.head_index]
                    if head_token.lemma.lower() != "column":
                        continue

                    # consider head token only if its not part of any table column
                    flag_stmnt_head_token_belongs_to_column = False

                    for col_index in selected_column_index_arr:
                        for row_index in column_matched_tokens_dict[col_index]:
                            for token_index_start_column, token_index_end_column in \
                            column_matched_tokens_dict[col_index][row_index]["token_index_range_statement"]:
                                if cur_token.head_index in range(token_index_start_column, token_index_end_column):
                                    flag_stmnt_head_token_belongs_to_column = True
                                    break

                            if flag_stmnt_head_token_belongs_to_column:
                                break

                        if flag_stmnt_head_token_belongs_to_column:
                            break

                    if flag_stmnt_head_token_belongs_to_column:
                        continue

                    flag_numeric, statement_token_value = is_number(cur_token.text)
                    flag_match = None
                    if flag_numeric:
                        if int(statement_token_value) == statement_token_value:
                            flag_match = statement_token_value == len(self.df.columns)
                    else:
                        # convert word to numeric value
                        try:
                            statement_token_value_numeric = w2n.word_to_num(statement_token_value)
                            flag_match = statement_token_value_numeric == len(self.df.columns)
                        except:
                            continue

                    if flag_match is True:
                        statement_type_predict = "entailed"
                    elif flag_match is False:
                        statement_type_predict = "refuted"

                    # assign evidence
                    if flag_match is not None:
                        for row_idx in range(self.table_data_start_row_index):
                            if row_idx not in self.cell_evidence_dict:
                                continue
                            for col_idx in range(len(self.df.columns)):
                                if col_idx in self.cell_evidence_dict[row_idx]:
                                    self.cell_evidence_dict[row_idx][col_idx].add(self.statements[stmnt_i].id)

                        if verbose:
                            print(
                                "\tNumber of columns in table: {} :: count mentioned in statement: {} :: match: {}".format(
                                    len(self.df.columns), cur_token.text, flag_match))

        # candidate: count rows for column
        if len(selected_column_index_arr) == 1 and statement_type_predict is None:
            col_index = selected_column_index_arr[0]

            if len(rows_matched) > 0:
                n_rows_matched_col = len(rows_matched)
            else:
                n_rows_matched_col = len(self.df.iloc[:, col_index].dropna())

            for token_index_stmnt in range(len(self.statements[stmnt_i].tokens)):
                cur_token = self.statements[stmnt_i].tokens[token_index_stmnt]
                if cur_token.part_of_speech_coarse == "NUM" and cur_token.dependency_tag == "nummod":
                    # consider only if its not part of the statement tokens matching the column, row
                    flag_stmnt_token_belongs_to_column = False

                    for row_index in column_matched_tokens_dict[col_index]:
                        for token_index_start_column, token_index_end_column in \
                        column_matched_tokens_dict[col_index][row_index]["token_index_range_statement"]:
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

                    flag_numeric, statement_token_value = is_number(cur_token.text)
                    flag_match = None
                    if flag_numeric:
                        # Basic check to skip considering float aa count
                        # TODO Need to identify text/dependency tree pattern to consider as candidate for count rows
                        if int(statement_token_value) == statement_token_value:
                            flag_match = statement_token_value == n_rows_matched_col
                    else:
                        # convert word to numeric value
                        try:
                            statement_token_value_numeric = w2n.word_to_num(statement_token_value)
                            flag_match = n_rows_matched_col == statement_token_value_numeric
                        except:
                            continue

                    if flag_match is True:
                        statement_type_predict = "entailed"
                    elif flag_match is False:
                        statement_type_predict = "refuted"

                    # assign evidence
                    if flag_match is not None:
                        for row_idx in range(self.table_data_end_row_index):
                            if row_idx in self.cell_evidence_dict and col_index in self.cell_evidence_dict[row_idx]:
                                self.cell_evidence_dict[row_idx][col_index].add(self.statements[stmnt_i].id)

                    if verbose:
                        print(
                            "\tNumber of rows matched for column: {} :: count mentioned in statement: {} :: match: {}".format(
                                n_rows_matched_col, statement_token_value, flag_match))

        # candidate:
        #   a) Whether multiple rows have same/different value for a column
        #   b) Whether multiple columns have same/different value for a row
        m = re.search(r'(\bsame\b|\bidentical\b|\bdifferent\b)', self.statements[stmnt_i].text, flags=re.I)

        if m and statement_type_predict is None:
            col_index = None
            candidate_cols = [x for x in selected_column_index_arr if x > 0]

            if len(candidate_cols) == 1 and len(rows_matched) > 1:
                # case: compare row values for a single column
                col_index = candidate_cols[0]

                cell_value_arr = [self.df.iloc[x[0] - self.table_data_start_row_index, col_index] for x in rows_matched]
                flag_candidate_identical = all([x == cell_value_arr[0] for x in cell_value_arr[1:]])

                if m.group(0).lower() in ["same", "identical"]:
                    if flag_candidate_identical:
                        statement_type_predict = "entailed"
                    else:
                        statement_type_predict = "refuted"
                else:
                    if flag_candidate_identical:
                        statement_type_predict = "refuted"
                    else:
                        statement_type_predict = "entailed"

                # assign evidence
                for row_info in rows_matched:
                    row_index = row_info[0]
                    if row_index in self.cell_evidence_dict:
                        if col_index in self.cell_evidence_dict[row_index]:
                            self.cell_evidence_dict[row_index][col_index].add(self.statements[stmnt_i].id)
                        # column representing the row names
                        if 0 in self.cell_evidence_dict[row_index]:
                            self.cell_evidence_dict[row_index][0].add(self.statements[stmnt_i].id)

                # column headers
                for row_idx in range(self.table_data_start_row_index):
                    if row_idx in self.cell_evidence_dict:
                        if col_index in self.cell_evidence_dict[row_idx]:
                            self.cell_evidence_dict[row_idx][col_index].add(self.statements[stmnt_i].id)
                        # column representing the row names
                        if 0 in self.cell_evidence_dict[row_idx]:
                            self.cell_evidence_dict[row_idx][0].add(self.statements[stmnt_i].id)

            elif len(candidate_cols) > 1 and len(rows_matched) == 1:
                # case: compare mentioned column values for a row
                row_index = rows_matched[0][0]

                cell_value_arr = [self.df.iloc[row_index - self.table_data_start_row_index, col_index] for col_index in
                                  candidate_cols]
                flag_candidate_identical = all([x == cell_value_arr[0] for x in cell_value_arr[1:]])

                if m.group(0).lower() in ["same", "identical"]:
                    if flag_candidate_identical:
                        statement_type_predict = "entailed"
                    else:
                        statement_type_predict = "refuted"
                else:
                    if flag_candidate_identical:
                        statement_type_predict = "refuted"
                    else:
                        statement_type_predict = "entailed"

                # assign evidence
                if row_index in self.cell_evidence_dict:
                    for col_idx in (candidate_cols + [0]):
                        if col_idx in self.cell_evidence_dict[row_index]:
                            self.cell_evidence_dict[row_index][col_idx].add(self.statements[stmnt_i].id)

                # assign evidence to column headers
                for row_idx in range(self.table_data_start_row_index):
                    if row_idx in self.cell_evidence_dict:
                        for col_idx in (candidate_cols + [0]):
                            if col_idx in self.cell_evidence_dict[row_idx]:
                                self.cell_evidence_dict[row_idx][col_idx].add(self.statements[stmnt_i].id)

        # candidate: comparative
        #   a) comparison between two columns
        #       i) comparison over each of the rows
        #       ii) comparison for a particular row
        #   b) comparison between two rows
        flag_comparative_greater_than = False
        flag_comparative_lesser_than = False
        comparative_token_index = None
        for token_i in range(len(self.statements[stmnt_i].tokens) - 1):
            if self.statements[stmnt_i].tokens[token_i + 1].text.lower() == "than":
                if self.statements[stmnt_i].tokens[token_i].text.lower() in ["greater", "larger", "higher", "bigger"]:
                    flag_comparative_greater_than = True
                    comparative_token_index = token_i
                    break
                elif self.statements[stmnt_i].tokens[token_i].text.lower() in ["smaller", "lower", "lesser"]:
                    flag_comparative_lesser_than = True
                    comparative_token_index = token_i
                    break

        if (flag_comparative_greater_than or flag_comparative_lesser_than) and statement_type_predict is None:
            if len(selected_column_index_arr) > 1:
                # identify the columns which are compared by the comparative
                col_info_arr = []
                for col_idx in selected_column_index_arr:
                    token_index_start_column = len(self.statements[stmnt_i].tokens)
                    token_index_end_column = -1

                    for row_idx in column_matched_tokens_dict[col_idx]:
                        token_index_start_column = min(token_index_start_column,
                                                       column_matched_tokens_dict[col_idx][row_idx]["token_index_range_statement"][0][0])
                        token_index_end_column = max(token_index_end_column,
                                                     column_matched_tokens_dict[col_idx][row_idx]["token_index_range_statement"][0][1])

                    col_info_arr.append((token_index_start_column, token_index_end_column, col_idx))

                col_info_arr = sorted(col_info_arr)

                # Now identify the columns which can be designated as LHS and RHS of comparative
                n_columns_matched = len(selected_column_index_arr)
                col_index_lhs_comparative = None
                col_index_rhs_comparative = None

                for col_i in range(n_columns_matched - 1):
                    cur_token_index_start_column = col_info_arr[col_i][0]
                    cur_token_index_end_column = col_info_arr[col_i][1]
                    next_token_index_start_column = col_info_arr[col_i + 1][0]
                    next_token_index_end_column = col_info_arr[col_i + 1][1]

                    if cur_token_index_end_column <= comparative_token_index < next_token_index_start_column:
                        col_index_lhs_comparative = col_info_arr[col_i][2]
                        col_index_rhs_comparative = col_info_arr[col_i + 1][2]
                        break
                    elif next_token_index_start_column < comparative_token_index < next_token_index_end_column:
                        # case: Can happen in case of multiIndex column in which the column headers of the column are written in disjoint form
                        # ?? Can there be a case when LHS of comparative contains the comparative_token_index
                        col_index_lhs_comparative = col_info_arr[col_i][2]
                        col_index_rhs_comparative = col_info_arr[col_i + 1][2]
                        break

                if col_index_lhs_comparative is not None:
                    if len(rows_matched) == 1:
                        row_index = rows_matched[0][0]
                        lhs_value = self.df.iloc[row_index - self.table_data_start_row_index, col_index_lhs_comparative]
                        rhs_value = self.df.iloc[row_index - self.table_data_start_row_index, col_index_rhs_comparative]

                        if is_numeric_dtype(lhs_value) and is_numeric_dtype(rhs_value):
                            flag_comparative_match = None

                            if flag_comparative_greater_than:
                                flag_comparative_match = lhs_value > rhs_value
                            elif flag_comparative_lesser_than:
                                flag_comparative_match = lhs_value < rhs_value

                            # converting numpy bool to bool
                            if bool(flag_comparative_match) is True:
                                statement_type_predict = "entailed"
                            elif bool(flag_comparative_match) is False:
                                statement_type_predict = "refuted"

                            # assign evidence
                            if flag_comparative_match is not None:
                                if row_index in self.cell_evidence_dict:
                                    for col_idx in [col_index_lhs_comparative, col_index_rhs_comparative, 0]:
                                        if col_idx in self.cell_evidence_dict[row_index]:
                                            self.cell_evidence_dict[row_index][col_idx].add(self.statements[stmnt_i].id)

                                # assign evidence to column headers
                                for row_idx in range(self.table_data_start_row_index):
                                    if row_idx in self.cell_evidence_dict:
                                        for col_idx in [col_index_lhs_comparative, col_index_rhs_comparative, 0]:
                                            if col_idx in self.cell_evidence_dict[row_idx]:
                                                self.cell_evidence_dict[row_idx][col_idx].add(
                                                    self.statements[stmnt_i].id)

                    elif len(rows_matched) == 0:
                        # case: compare each element of the two columns
                        m = re.search(r'(\beach\b|\bevery\b)', self.statements[stmnt_i].text, flags=re.I)
                        flag_comparative_match = None
                        if m and is_numeric_dtype(self.df.iloc[:, col_index_lhs_comparative]) and \
                                is_numeric_dtype(self.df.iloc[:, col_index_rhs_comparative]):
                            if flag_comparative_greater_than:
                                flag_comparative_match = all(self.df.iloc[:, col_index_lhs_comparative] > self.df.iloc[:, col_index_rhs_comparative])
                            elif flag_comparative_lesser_than:
                                flag_comparative_match = all(self.df.iloc[:, col_index_lhs_comparative] < self.df.iloc[:, col_index_rhs_comparative])

                        if flag_comparative_match is True:
                            statement_type_predict = "entailed"
                        elif flag_comparative_match is False:
                            statement_type_predict = "refuted"

                        # assign evidence
                        if flag_comparative_match is not None:
                            for col_idx in [col_index_lhs_comparative, col_index_rhs_comparative]:
                                for row_idx in range(self.table_data_end_row_index):
                                    if row_idx in self.cell_evidence_dict and col_idx in self.cell_evidence_dict[
                                        row_idx]:
                                        self.cell_evidence_dict[row_idx][col_idx].add(self.statements[stmnt_i].id)

            elif len(rows_matched) > 1:
                pass

        # candidate: cell value
        if len(rows_matched) == 1 and statement_type_predict is None:
            numeric_column_index_arr = []
            non_numeric_column_index_arr = []

            for c_index in sorted(selected_column_index_arr):
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
                    match_token_length = len(self.cell_info_dict[row_index][col_index]["tokens"])

                    flag_cell_matched = None
                    for token_index_stmnt in range(len(self.statements[stmnt_i].tokens) - match_token_length + 1):
                        token_i = 0
                        flag_cell_matched = True
                        while token_i < match_token_length:
                            if (self.statements[stmnt_i].tokens[token_index_stmnt + token_i].text.lower() !=
                                    self.cell_info_dict[row_index][col_index]["tokens"][token_i].text.lower()) and \
                                    (self.statements[stmnt_i].tokens[token_index_stmnt + token_i].lemma.lower() !=
                                         self.cell_info_dict[row_index][col_index]["tokens"][token_i].lemma.lower()):
                                flag_cell_matched = False
                                break
                            token_i += 1

                        if flag_cell_matched:
                            if verbose:
                                cell_text = " ".join(
                                    [self.cell_info_dict[row_index][col_index]["tokens"][i].text for i in
                                     range(len(self.cell_info_dict[row_index][col_index]["tokens"]))])
                                print("\tCell matched: {}".format(cell_text))

                            break

                    if flag_cell_matched:
                        statement_type_predict = "entailed"
                    elif not flag_cell_matched:
                        statement_type_predict = "refuted"

                    # evidence
                    for col_idx in [col_index, 0]:
                        if row_index in self.cell_evidence_dict and col_idx in self.cell_evidence_dict[row_index]:
                            self.cell_evidence_dict[row_index][col_idx].add(self.statements[stmnt_i].id)

                        # column headers
                        for row_idx in range(self.table_data_start_row_index):
                            if row_idx in self.cell_evidence_dict and col_idx in self.cell_evidence_dict[row_idx]:
                                self.cell_evidence_dict[row_idx][col_idx].add(self.statements[stmnt_i].id)

        return statement_type_predict

    def build_table_element(self, ref_table_elem, statement_id_predict_type_map):
        """Build XML table element with predictions.
            Populate only the fields which are absolutely required for evaluation.

            Parameters
            ----------
            ref_table_elem : xml.etree.ElementTree.Element
                Table xml element in input xml.
            statement_id_predict_type_map : dict
                key: statement id  value: type prediction

            Returns
            -------
            table xml element

            Note
            ----
            There are instances of missing statements in evidence. e.g. 10232.xml, Table 7
            This leads to crashes in evaluate script. Hence considering only the statements which are mentioned in evidence element in ref_table_elem.
        """
        pred_table_elem = ET.Element("table")
        pred_table_elem.set("id", self.table_id)

        for ref_row_elem in ref_table_elem.iterfind("row"):
            row_index = int(ref_row_elem.attrib["row"])
            # assert row_index in self.cell_evidence_dict, "row_index: {} absent in cell_evidence_dict".format(row_index)
            pred_row_elem = ET.Element("row")
            pred_row_elem.set("row", str(row_index))
            for ref_cell_elem in ref_row_elem.iterfind("cell"):
                col_index = int(ref_cell_elem.attrib["col-start"])
                row_index = int(ref_cell_elem.attrib["row-start"])
                # assert col_index in self.cell_evidence_dict[row_index], "col_index: {} absent in cell_evidence_dict corresponding to row_index: {}".format(col_index, row_index)

                pred_cell_elem = ET.SubElement(pred_row_elem, "cell")
                pred_cell_elem.set("col-end", ref_cell_elem.attrib["col-end"])
                pred_cell_elem.set("col-start", ref_cell_elem.attrib["col-start"])
                pred_cell_elem.set("row-end", ref_cell_elem.attrib["row-end"])
                pred_cell_elem.set("row-start", ref_cell_elem.attrib["row-start"])

                for ref_evidence_elem in ref_cell_elem.iterfind("evidence"):
                    statement_id = ref_evidence_elem.attrib["statement_id"]

                    pred_evidence_elem = ET.SubElement(pred_cell_elem, "evidence")
                    pred_evidence_elem.set("statement_id", statement_id)

                    if row_index not in self.cell_evidence_dict or col_index not in self.cell_evidence_dict[row_index]:
                        # case: empty cell
                        pred_evidence_elem.set("type", "irrelevant")
                    elif statement_id in self.cell_evidence_dict[row_index][col_index]:
                        pred_evidence_elem.set("type", "relevant")
                    else:
                        pred_evidence_elem.set("type", "irrelevant")

                    version = ref_evidence_elem.attrib["version"]
                    if version == "":
                        version = "0"
                    pred_evidence_elem.set("version", version)

            pred_table_elem.append(pred_row_elem)

        # TODO Populate statements element based on ref_table_elem
        pred_statements_elem = ET.Element("statements")
        pred_table_elem.append(pred_statements_elem)

        for stmnt_i in range(len(self.statements)):
            statement_elem = ET.SubElement(pred_statements_elem, "statement")
            statement_elem.set("id", self.statements[stmnt_i].id)
            # statement_elem.set("text", self.statements[stmnt_i].text)
            if self.statements[stmnt_i].id in statement_id_predict_type_map:
                statement_type_predict = statement_id_predict_type_map[self.statements[stmnt_i].id]
            else:
                statement_type_predict = None

            if statement_type_predict is None:
                statement_type_predict = "unknown"

            statement_elem.set("type", statement_type_predict)

        return pred_table_elem

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
                        min_column_value = numeric_column_value
                        min_column_value_df_row_index = df_row_index

                    if max_column_value < numeric_column_value:
                        max_column_value = numeric_column_value
                        max_column_value_df_row_index = df_row_index

        return min_column_value, min_column_value_df_row_index,  max_column_value, max_column_value_df_row_index

    def select_matched_columns(self, stmnt_i, column_matched_tokens_dict):
        """Select columns from the list of matched column candidates.
            Approximate string matching allows multiple columns to match subtext of statements.
            Here we use matching score to prepare final list of matched columns.
        """
        selected_column_index_arr = []

        if len(column_matched_tokens_dict) == 0:
            return selected_column_index_arr

        if len(column_matched_tokens_dict) == 1:
            return list(column_matched_tokens_dict.keys())

        col_index_to_avg_score_map = dict()
        for col_index in column_matched_tokens_dict:
            avg_score = 0
            token_index_start = len(self.statements[stmnt_i].tokens)
            token_index_end = -1
            for row_index in column_matched_tokens_dict[col_index]:
                avg_score += column_matched_tokens_dict[col_index][row_index]["score"][0]
                token_index_start = min(token_index_start, column_matched_tokens_dict[col_index][row_index]["token_index_range_statement"][0][0])
                token_index_end = max(token_index_end, column_matched_tokens_dict[col_index][row_index]["token_index_range_statement"][0][1])

            avg_score /= len(column_matched_tokens_dict[col_index].keys())
            col_index_to_avg_score_map[col_index] = (-1*avg_score, token_index_start, -1*token_index_end)

        candidate_col_index_arr = [k for k, v in sorted(col_index_to_avg_score_map.items(), key=lambda x: x[1])]

        selected_column_index_arr.append(candidate_col_index_arr[0])
        for cur_col_index in candidate_col_index_arr[1:]:
            # select cur_col_index only if it has no overlap with other selected columns
            flag_overlap = False
            for prev_col_index in selected_column_index_arr:
                for row_index in column_matched_tokens_dict[cur_col_index]:
                    if row_index not in column_matched_tokens_dict[prev_col_index]:
                        continue
                    cur_col_token_index_range_statement = column_matched_tokens_dict[cur_col_index][row_index]["token_index_range_statement"][0]
                    prev_col_token_index_range_statement = column_matched_tokens_dict[prev_col_index][row_index]["token_index_range_statement"][0]

                    if (cur_col_token_index_range_statement[0] >= prev_col_token_index_range_statement[1]) or \
                            (cur_col_token_index_range_statement[1] <= prev_col_token_index_range_statement[0]):
                        pass
                    else:
                        flag_overlap = True
                        break

                if flag_overlap:
                    break

            if flag_overlap is False:
                selected_column_index_arr.append(cur_col_index)

        return selected_column_index_arr
