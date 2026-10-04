from typing import List


class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        cnt_0 = [0] * (n + 1)
        cnt_1 = [0] * (n + 1)
        res = 0
        i = 0
        while i < n:
            if s[i] in 'aeiou':
                cnt_0[i + 1] = cnt_0[i] + 1
                cnt_1[i + 1] = cnt_1[i]
            else:
                cnt_0[i + 1] = cnt_0[i]
                cnt_1[i + 1] = cnt_1[i] + 1
            i += 1
        i = 0
        while i < n:
            j = i
            while j <= n:
                v1 = cnt_0[j] - cnt_0[i]
                v2 = cnt_1[j] - cnt_1[i]
                if v1 == v2 and (v1 * v2) % k == 0:
                    res += 1
                j += 1
            i += 1
        return res
