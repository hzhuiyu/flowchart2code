class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        count = 0
        for num in range(low, high + 1):
            even = 0
            odd = 0
            for d in str(num):
                if int(d) % 2 == 0:
                    even += 1
                else:
                    odd += 1
            if even == odd or num % k == 0:
                count += 1
        return count
