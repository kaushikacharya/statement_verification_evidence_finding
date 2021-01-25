#!/usr/bin/env python

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
