from typing import List


class Solution:
    def countBlackBlocks(self, m: int, n: int, coordinates: List[List[int]]) -> List[int]:
        ans = [0, 0, 0, 0, 0]
        cnt = {}
        for c in coordinates:
            for x in (c[0] - 1, c[0]):
                if 0 <= x <= m - 1:
                    for y in (c[1] - 1, c[1]):
                        if 0 <= y <= n - 1:
                            cnt[(x, y)] = cnt.get((x, y), 0) + 1
        for v in cnt.values():
            ans[v] += 1
        ans[0] = (m - 1) * (n - 1) - (ans[1] + ans[2] + ans[3] + ans[4])
        return ans
