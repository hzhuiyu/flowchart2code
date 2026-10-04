from typing import List


class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        sum_vals = [0]
        for v in nums:
            sum_vals.append(sum_vals[-1] + v)

        left = 0
        best = 1
        right = 0
        while right < n:
            while True:
                mid = (left + right) // 2
                val = (sum_vals[right + 1] - sum_vals[mid + 1]) - (right - mid) * nums[mid]
                val += (mid - left) * nums[mid] - (sum_vals[mid] - sum_vals[left])
                if val > k:
                    left += 1
                else:
                    break
            best = max(best, right - left)
            right += 1
        return best
