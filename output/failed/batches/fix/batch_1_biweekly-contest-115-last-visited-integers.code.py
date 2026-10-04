from typing import List


class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        ov = []
        ans = []
        ci = -1
        for s in words:
            if s == 'prev':
                if ci < 0:
                    ans.append(-1)
                else:
                    ci -= 1
                    ans.append(ov[ci])
            else:
                vv = 0
                for c in s:
                    vv = vv * 10 + int(c)
                ov.append(vv)
                ci = len(ov)

        return ans
