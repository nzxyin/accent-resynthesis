from typing import Tuple

# _vowel_set = {
#     "AA",
#     "AA0",
#     "AA1",
#     "AA2",
#     "AE",
#     "AE0",
#     "AE1",
#     "AE2",
#     "AH",
#     "AH0",
#     "AH1",
#     "AH2",
#     "AO",
#     "AO0",
#     "AO1",
#     "AO2",
#     "AW",
#     "AW0",
#     "AW1",
#     "AW2",
#     "AY",
#     "AY0",
#     "AY1",
#     "AY2",
#     "EH",
#     "EH0",
#     "EH1",
#     "EH2",
#     "ER",
#     "ER0",
#     "ER1",
#     "ER2",
#     "EY",
#     "EY0",
#     "EY1",
#     "EY2",
#     "IH",
#     "IH0",
#     "IH1",
#     "IH2",
#     "IY",
#     "IY0",
#     "IY1",
#     "IY2",
#     "OW",
#     "OW0",
#     "OW1",
#     "OW2",
#     "OY",
#     "OY0",
#     "OY1",
#     "OY2",
#     "UH",
#     "UH0",
#     "UH1",
#     "UH2",
#     "UW",
#     "UW0",
#     "UW1",
#     "UW2",
# }

# _cons_set = {
#     "B",
#     "CH",
#     "D",
#     "DH",
#     "F",
#     "G",
#     "HH",
#     "JH",
#     "K",
#     "L",
#     "M",
#     "N",
#     "NG",
#     "P",
#     "R",
#     "S",
#     "SH",
#     "T",
#     "TH",
#     "V",
#     "W",
#     "Y",
#     "Z",
#     "ZH",
# }

# _arpa = list(_vowel_set + _cons_set).sort()

_pad = "_"

_silences = ["<bos>", "<eos>", "<sp>"]

# symbols = [_pad] + _arpa + _silences

_unilex = ['*@', '*@@r', '*a', '*aa', '*ae', '*aer',
 '*ah', '*ai', '*ao', '*ar', '*au', '*e', '*ee', '*ei', '*eir',
 '*er', '*i', '*i@', '*ii', '*ir', '*iu', '*o', '*oa', '*oi', '*oir',
 '*oo', '*oou', '*or', '*ou', '*our', '*ouw', '*ow', '*owr', '*u', '*uh',
 '*ur', '*uu', '@', '@@r', '@r', 'a', 'aa', 'ae', 'aer', 'ah', 'ai',
 'ao', 'ar', 'au', 'b', 'ch', 'd', 'dh', 'e', 'ee', 'ei', 'eir', 'er',
 'f', 'g', 'h', 'hw', 'i', 'i@', 'ii', 'ir', 'iu', 'iy', 'jh', 'k', 'l',
 'll', 'm', 'n', 'ng', 'o', 'oa', 'oi', 'oo', 'oou', 'or', 'ou', 'our',
 'ouw', 'ow', 'owr', 'p', 'r', 's', 'sh', 't', 'th', 'u', 'uh', 'ur',
 'uu', 'v', 'w', 'x', 'y', 'z', 'zh']

symbols = [_pad] + _silences + _unilex

_symbol_to_id = {symbols[i]: i for i in range(len(symbols))}

_id_to_symbol = {i: symbols[i] for i in range(len(symbols))}

def get_id(symbol: str) -> int:
    return _symbol_to_id[symbol]

def get_symbol(id: int) -> str:
    return _id_to_symbol[id]