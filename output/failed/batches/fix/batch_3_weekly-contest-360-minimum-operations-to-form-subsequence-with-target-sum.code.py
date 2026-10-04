from typing import List


class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        if sum(nums) < target:
            return -1
        cnt = [0] * 32
        for v in nums:
            cnt[v.bit_length() - 1] += 1
        ops = 0
        for i in range(31):
            if (target >> i) & 1:
                if cnt[i] > 0:
                    cnt[i] -= 1
                else:
                    j = i + 1
                    while j < 32 and cnt[j] == 0:
                        j += 1
                    if j == 32:
                        return -1
                    ops += j - i - 1
                    cnt[j] -= 1
                    for p in range(i, j):
                        cnt[p] += 1
                    cnt[i] -= 1
            cnt[i + 1] += cnt[i] // 2
        return ops
