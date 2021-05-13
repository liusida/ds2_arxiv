from difflib import SequenceMatcher
import unicodedata

def to_English(s):
    try:
        s = unicodedata.normalize('NFKD', s).encode('latin1', 'ignore').decode('utf8')
    finally:
        return s

def similar(name1, name2):
    ret = False
    _r = SequenceMatcher(a=name1, b=name2).ratio()
    if _r>0.8:
        ret = True
    return ret

def compare_two_names(name1, name2):
    """ compare two names in different ways.
    return True if two names refer to the same author. """
    ret = False

    # Using English alphabet. e.g. treat Ž as Z
    name1, name2 = to_English(name1), to_English(name2)

    # Whole names are similar, tolerant to typos.
    if similar(name1, name2):
        ret = True

    _name1 = name1.split(" ")
    _name2 = name2.split(" ")
    if len(_name1)>1 and len(_name2)>1:
        # First Name and Last Name are the same
        if similar(_name1[0], _name2[0]) and similar(_name1[-1], _name2[-1]):
            ret = True
        # First Initial and Last Initial are the same
        if f"{_name1[0][0]}." == f"{_name2[0][0]}." and f"{_name1[-1][0]}." == f"{_name2[-1][0]}.":
            ret = True
        # First Initial and Last Initial are the same, but wrong order
        if f"{_name1[0][0]}." == f"{_name2[-1][0]}." and f"{_name1[-1][0]}." == f"{_name2[0][0]}.":
            ret = True
        # Middle Initial and Last Initial are the same
        if f"{_name1[-2][0]}." == f"{_name2[-2][0]}." and f"{_name1[-1][0]}." == f"{_name2[-1][0]}.":
            ret = True
        # First Initial and Second Initial are the same
        if f"{_name1[0][0]}." == f"{_name2[0][0]}." and f"{_name1[1][0]}." == f"{_name2[1][0]}.":
            ret = True

    return ret

if __name__=="__main__":

    def unittest():
        same_list = [
            ("Kevin Yang", "Kevin Yang"),
            ("Kevin Yang", "Kevin F. Yang"),
            ("Kevin Yang", "K. F. Yang"),
            ("Weinan E", "E. Weinan"),
            ("Jure Žbontar", "J. Zbontar"),
        ]
        diff_list = [
            ("Kevin Yang", "John W. Paisley"),
        ]
        for p in same_list:
            assert compare_two_names(*p), f"fail on {p}"
        for p in diff_list:
            assert not compare_two_names(*p), f"fail on {p}"

    unittest()