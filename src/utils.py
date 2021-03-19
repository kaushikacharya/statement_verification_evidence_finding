#!/usr/bin/env python

from math import floor
import re
from xml.dom import minidom
import xml.etree.ElementTree as ET

def is_number(s):
    """Identify whether input string is a number or not.

    References
    ----------
    https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
        General consensus is on the try catch method being the fastest method.
    https://stackoverflow.com/questions/43156077/how-to-check-a-string-is-float-or-int
        YCF_L's solution 2 suggests regex. But it doesn't cover the exponential part.
    https://www.geeksforgeeks.org/check-given-string-valid-number-integer-floating-point-java-set-2-regular-expression-approach/
        Regex handles exponential part too.
    """
    try:
        # minus sign in negative numbers have been found in the dataset with minus sign: unicode U+2212
        # These are replaced with the regular hyphen minus symbol unicode U+002D
        # https://unicodelookup.com/#minus/1 mentions both the symbols.
        if s.startswith("−"):
            s = "-" + s[1:]

        return True, float(s)
    except ValueError:
        return False, s

def modify_text(s):
    """Replace unexpected characters with expected characters

        Examples
        --------
        10257.xml, Table 4
            Table cell contains: A-A′  (last character: prime) ((U+2032)
                            https://www.fileformat.info/info/unicode/char/2032/index.htm
                            https://unicodelookup.com/#prime/1
            Whereas statement contains: A-A' (last character: apostrophe) ((U+0027)
                            https://www.fileformat.info/info/unicode/char/27/index.htm
                            https://unicodelookup.com/#apostrophe/1

        Note
        ----
        Foot is denoted by prime symbol. (https://en.wikipedia.org/wiki/Foot_(unit)#Symbol)
        Hence replacing it with apostrophe may not be a correct approach.
    """
    # replace prime character by apostrophe
    s = re.sub(r"′", r"'", s)

    return s

def prettify(elem):
    """Return a pretty-printed XML ElementTree for the Element.

        References
        ----------
        https://stackoverflow.com/questions/17402323/use-xml-etree-elementtree-to-print-nicely-formatted-xml-files
            - Maxime Chéramy's answer
        https://stackoverflow.com/questions/51660316/attributeerror-elementtree-object-has-no-attribute-tag-in-python
            - Madhan Varadhodiyil's answer for handling AttributeError
        https://stackoverflow.com/questions/647071/python-xml-elementtree-from-a-string-source
            - dgassaway's answer to convert Element into ElementTree
    """
    root = elem.getroot()
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return ET.ElementTree(ET.fromstring(reparsed.toprettyxml(indent="\t")))


def jaro_similarity(s1, s2):
    """Jaro Similarity

        Parameters
        ----------
        s1 : str
        s2 : str

        References
        ----------
        https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance
        https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity/
            - Implementation
    """

    # If the strings are equal
    if s1 == s2:
        return 1.0

    if s1 == "" or s2 == "":
        return 0.0

    # Length of two strings
    len1 = len(s1)
    len2 = len(s2)

    # Maximum distance up to which matching is allowed.
    # Two characters from s1 and s2 are considered for matching only if its within max_dist characters away.
    max_dist = (max(len(s1), len(s2)) // 2) - 1

    # Count of matches
    match = 0

    # Hash for matches
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)

    # Traverse through the first string
    for i in range(len1):
        # Check if there is any matches
        for j in range(max(0, i - max_dist), min(len2, i + max_dist + 1)):
            # If there is a match
            if (s1[i] == s2[j]) and (hash_s2[j] == 0):
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break

    # If there is no match
    if match == 0:
        return 0.0

    # Number of transpositions
    t = 0
    point = 0

    # Count number of occurrences
    # where two characters match but
    # there is a third matched character
    # in between the indices
    for i in range(len1):
        if hash_s1[i]:
            # Find the next matched character
            # in second string
            while hash_s2[point] == 0:
                point += 1

            if s1[i] != s2[point]:
                point += 1
                t += 1
            else:
                point += 1

        t /= 2

    # Return the Jaro Similarity
    return (match / len1 + match / len2 + (match - t) / match) / 3.0

def jaro_winkler_similarity(s1, s2, jaro_sim, max_prefix_length=4, prefix_scaling_factor=0.1):
    """Jaro-Winkler similarity

        Parameters
        ----------
        s1 : str
        s2 : str
        jaro_sim : float (Jaro similarity)
        max_prefix_length : int (Max prefix length considered)
        prefix_scaling_factor : float
    """

    # Find the common prefix length
    prefix_len = 0

    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    prefix_len = min(max_prefix_length, prefix_len)

    jaro_winkler_sim = jaro_sim + prefix_len*prefix_scaling_factor*(1 - jaro_sim)

    return jaro_winkler_sim
