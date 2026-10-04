from typing import List


class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        ans = 0
        i = 0
        while i < n:
            # A valid subarray must start on an even element within threshold.
            if nums[i] % 2 == 0 and nums[i] <= threshold:
                j = i + 1
                # Extend while elements stay within threshold and parity alternates.
                while j < n and nums[j] <= threshold and nums[j] % 2 != nums[j - 1] % 2:
                    j += 1
                ans = max(ans, j - i + 1)
                i = j
            else:
                i += 1
        return ans
