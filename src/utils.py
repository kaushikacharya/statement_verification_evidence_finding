#!/usr/bin/env python

def is_number(s):
    """Identify whether input string is a number or not

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
        float(s)
        return True
    except ValueError:
        return False
