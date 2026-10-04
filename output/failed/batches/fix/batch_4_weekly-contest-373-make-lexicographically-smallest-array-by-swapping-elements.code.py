from typing import List


class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        pairs = sorted(zip(nums, range(n)))
        groups = []
        result = [0] * n
        i = 0
        while i < n:
            if i == 0 or pairs[i][0] - pairs[i - 1][0] >= limit:
                groups.append([])
            groups[-1].append(pairs[i])
            i += 1
        g = 0
        while g < len(groups):
            idxs = sorted([p[1] for p in groups[g]])
            k = 0
            while k < len(idxs):
                result[idxs[k]] = groups[g][k][0]
                k += 1
            g += 1
        return result
