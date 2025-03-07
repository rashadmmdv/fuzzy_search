import numpy as np
from typing import List, Dict

def load_words(filename: str) -> List[str]:
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        print(f"Warning: File '{filename}' not found.")
        return []

def lev_recursive(s1: str, s2: str) -> int:
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    if s1[-1] == s2[-1]:
        return lev_recursive(s1[:-1], s2[:-1])
    return 1 + min(
        lev_recursive(s1[:-1], s2),
        lev_recursive(s1, s2[:-1]),
        lev_recursive(s1[:-1], s2[:-1])
    )

def lev_dynamic(s1: str, s2: str) -> int:
    len_s1, len_s2 = len(s1), len(s2)
    dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)
    
    for i in range(len_s1 + 1):
        for j in range(len_s2 + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return int(dp[len_s1][len_s2])

def lev_optimized(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def jaro_winkler(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    
    match_distance = max(len(s1), len(s2)) // 2 - 1
    match_distance = max(0, match_distance)  
    
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    
    matching_chars = 0
    for i in range(len(s1)):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(s2))
        
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matching_chars += 1
                break
    
    if matching_chars == 0:
        return 0.0
    
    transpositions = 0
    j = 0
    for i in range(len(s1)):
        if s1_matches[i]:
            while not s2_matches[j]:
                j += 1
            if s1[i] != s2[j]:
                transpositions += 1
            j += 1
    
    transpositions = transpositions // 2
    jaro_similarity = (matching_chars / len(s1) + 
                       matching_chars / len(s2) + 
                       (matching_chars - transpositions) / matching_chars) / 3
    
    prefix_length = 0
    max_prefix_length = min(4, min(len(s1), len(s2)))
    for i in range(max_prefix_length):
        if s1[i] == s2[i]:
            prefix_length += 1
        else:
            break
    
    p = 0.1
    jaro_winkler_similarity = jaro_similarity + (prefix_length * p * (1 - jaro_similarity))
    
    return jaro_winkler_similarity

def damerau_levenshtein(s1: str, s2: str) -> int:
    len_s1, len_s2 = len(s1), len(s2)
    dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)
    
    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j
    
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      
                dp[i][j - 1] + 1,       
                dp[i - 1][j - 1] + cost 
            )
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost)
    
    return dp[len_s1][len_s2]

def find_lcs(seq1: str, seq2: str) -> str:
    len_seq1, len_seq2 = len(seq1), len(seq2)
    dp = [["" for _ in range(len_seq2 + 1)] for _ in range(len_seq1 + 1)]

    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + seq1[i - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], key=len)

    return dp[len_seq1][len_seq2]

def find_top_5_matches(input_word: str, method: str, words_dataset: List[str]) -> List[Dict[str, any]]:
    word_details = []
    for word in words_dataset:
        if method == 'jaro_winkler':
            similarity = jaro_winkler(input_word, word)
            word_details.append({
                "word": word,
                "similarity": round(similarity, 4),
                "similarity_percentage": round(similarity * 100, 2)
            })
            word_details.sort(key=lambda x: x['similarity'], reverse=True)
        else:
            if method == 'recursive':
                distance = lev_recursive(input_word, word)
            elif method == 'optimized':
                distance = lev_optimized(input_word, word)
            elif method == 'damerau_levenshtein':
                distance = damerau_levenshtein(input_word, word)
            else: 
                distance = lev_dynamic(input_word, word)
                
            max_length = max(len(input_word), len(word))
            similarity_percentage = ((max_length - distance) / max_length) * 100
            
            word_details.append({
                "word": word,
                "distance": int(distance),
                "similarity_percentage": round(similarity_percentage, 2)
            })
            word_details.sort(key=lambda x: x['distance'])
    
    return word_details[:5]