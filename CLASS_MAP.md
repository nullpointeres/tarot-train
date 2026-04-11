# Tarot Card Class ID Mapping (card_no: 0-77)
# 0: The Fool         22: The Fool         45: Wands 6
# 1: The Magician     23: Wands Ace        46: Wands 7
# 2: The High Priestess 24: Wands 2         47: Wands 8
# 3: The Empress      25: Wands 3         48: Wands 9
# 4: The Emperor      26: Wands 4         49: Wands 10
# 5: The Hierophant    27: Wands 5         50: Wands Page
# 6: The Lovers       28: Wands 6         51: Wands Knight
# 7: The Chariot      29: Wands 7         52: Wands Queen
# 8: Strength         30: Wands 8         53: Wands King
# 9: The Hermit       31: Wands 9         54: Cups Ace
# 10: Wheel of Fortune 32: Wands 10        55: Cups 2
# 11: Justice         33: Wands Page       56: Cups 3
# 12: The Hanged Man   34: Wands Knight    57: Cups 4
# 13: Death           35: Wands Queen      58: Cups 5
# 14: Temperance      36: Wands King       59: Cups 6
# 15: The Devil       37: Cups Ace         60: Cups 7
# 16: The Tower       38: Cups 2           61: Cups 8
# 17: The Star        39: Cups 3           62: Cups 9
# 18: The Sun         40: Cups 4           63: Cups 10
# 19: Judgment        41: Cups 5           64: Cups Page
# 20: The World       42: Cups 6            65: Cups Knight
# 21: (blank)         43: Cups 7            66: Cups Queen
# (22 is actually The Fool - corrected below)
#
# Correct card_no mapping:
# card_no 0-21  = Major Arcana T-00 ~ T-21
# card_no 22-36  = Wands   W-0A(1), W-02~W-10, W-J1, W-J2, W-KI, W-QU (ACE=22)
# card_no 37-51  = Cups    C-0A(1), C-02~C-10, C-J1, C-J2, C-KI, C-QU
# card_no 52-66  = Swords  S-0A(1), S-02~S-10, S-J1, S-J2, S-KI, S-QU
# card_no 67-77  = Pentacles P-0A(1), P-02~P-10, P-J1, P-J2, P-KI (missing QU, has 11)
