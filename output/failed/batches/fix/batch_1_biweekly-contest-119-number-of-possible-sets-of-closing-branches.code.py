from typing import List


class Solution:
    def numberOfSets(self, n: int, maxDistance: int, roads: List[List[int]]) -> int:
        ans = 0
        for mask in range(1 << n):
            g = [[float('inf')] * n for _ in range(n)]
            for a in range(n):
                g[a][a] = 0
            for r in roads:
                if (mask >> r[0] & 1) == 0 and (mask >> r[1] & 1) == 0:
                    if r[2] < g[r[0]][r[1]]:
                        g[r[0]][r[1]] = r[2]
                        g[r[1]][r[0]] = r[2]
            for k in range(n):
                for a in range(n):
                    for b in range(n):
                        if g[a][k] + g[k][b] < g[a][b]:
                            g[a][b] = g[a][k] + g[k][b]
            ok = 1
            for a in range(n):
                for b in range(n):
                    if (mask >> a & 1) == 0 and (mask >> b & 1) == 0 and g[a][b] >= maxDistance:
                        ok = 0
            if ok == 1:
                ans += 1
        return ans
