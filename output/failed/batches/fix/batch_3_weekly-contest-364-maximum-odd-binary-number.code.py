class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        c = s.count('1')
        res = ''
        while len(res) < len(s):
            if c > 1:
                res += '1'
                c -= 1
            else:
                res += '0'
        return res + '1'
