class Solution:
    def finalString(self, s: str) -> str:
        result = []
        for char in s:
            if char == 'i':
                # The faulty key reverses what has been typed so far.
                result.append(char)
                result.reverse()
            else:
                result.append(char)
        return ''.join(result)
