def levenshtein(str1: str, str2: str) -> int:
    """Calculates the levenshtein distance between two strings."""
    if len(str1) < len(str2):
        return levenshtein(str2, str1)
    if len(str2) == 0:
        return len(str1)
    previous_row = range(len(str2) + 1)
    for i, c1 in enumerate(str1):
        current_row = [i + 1]
        for j, c2 in enumerate(str2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def custom_levenshtein(str1: str, str2: str) -> int:
    """
    Performs the levenshtein distance, but with lower cost for the following operations:
    - substitution of 7 by Z
    - substitution of 1 by I
    - substitution of 0 by O
    - substitution of B by 3
    """
    if len(str1) < len(str2):
        return custom_levenshtein(str2, str1)
    if len(str2) == 0:
        return len(str1)
    previous_row = range(len(str2) + 1)
    for i, c1 in enumerate(str1):
        current_row = [i + 1]
        if c1 == '7':
            c1 = 'Z'
        elif c1 == '1':
            c1 = 'I'
        elif c1 == '0':
            c1 = 'O'
        elif c1 == 'B':
            c1 = '3'
        for j, c2 in enumerate(str2):
            if c2 == '7':
                c2 = 'Z'
            elif c2 == '1':
                c2 = 'I'
            elif c2 == '0':
                c2 = 'O'
            elif c2 == 'B':
                c2 = '3'
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

if __name__ == "__main__":
    # take 2 positional arguments
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 levenshtein.py <string1> <string2>")
        exit(1)

    str1 = sys.argv[1]
    str2 = sys.argv[2]
    print(levenshtein(str1, str2))