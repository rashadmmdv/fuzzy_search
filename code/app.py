from fastapi import FastAPI, HTTPException
from typing import List, Dict, Literal
import time

from utils import (
    load_words,
    lev_recursive,
    lev_dynamic,
    lev_optimized,
    jaro_winkler,
    damerau_levenshtein,
    find_lcs,
    find_top_5_matches
)

app = FastAPI()
try:
    words_dataset = load_words("words.txt")
except Exception as e:
    print(f"Warning: Could not load words dataset: {e}")
    words_dataset = []

@app.get("/search/")
def search_word(word: str, method: Literal['dynamic', 'optimized', 'jaro_winkler', 'damerau_levenshtein'] = 'dynamic'):  
    
    start_time = time.time()
    results = find_top_5_matches(word, method, words_dataset)
    elapsed_time = time.time() - start_time

    if method == 'jaro_winkler':
        return {
            "input": word,
            "method": method,
            "metric_type": "similarity",
            "top_matches": results,
            "time_taken": round(elapsed_time, 4)
        }
    else:
        return {
            "input": word,
            "method": method,
            "metric_type": "distance",
            "top_matches": results,
            "time_taken": round(elapsed_time, 4)
        }
    
@app.get("/search-full-comparison/")
def api_search_full_comparison(word: str, compare_with: str = None):
    if not word:
        raise HTTPException(status_code=400, detail="Search word must be provided.")
    
    methods = ['dynamic', 'optimized', 'jaro_winkler', 'damerau_levenshtein']
    
    if compare_with:
        max_length = max(len(word), len(compare_with))
        results = {}
        
        for method in methods:
            start_time = time.time()
            
            if method == 'jaro_winkler':
                similarity = jaro_winkler(word, compare_with)
                results[method] = {
                    "similarity": round(similarity, 4),
                    "similarity_percentage": round(similarity * 100, 2),
                    "metric_type": "similarity",
                    "time_taken": round(time.time() - start_time, 4)
                }
            else:
                if method == 'dynamic':
                    distance = lev_dynamic(word, compare_with)
                elif method == 'optimized':
                    distance = lev_optimized(word, compare_with)
                elif method == 'damerau_levenshtein':
                    distance = damerau_levenshtein(word, compare_with)
                    
                similarity_percentage = ((max_length - distance) / max_length) * 100
                results[method] = {
                    "distance": int(distance),
                    "similarity_percentage": round(similarity_percentage, 2),
                    "metric_type": "distance",
                    "time_taken": round(time.time() - start_time, 4)
                }
        
        return {
            "input": word,
            "compared_with": compare_with,
            "input_length": len(word),
            "compared_with_length": len(compare_with),
            "comparison_mode": "direct",
            "algorithm_results": results
        }
    
    else:
        results = {}
        
        for method in methods:
            start_time = time.time()
            top_matches = find_top_5_matches(word, method, words_dataset)
            elapsed_time = time.time() - start_time
            
            if method == 'jaro_winkler':
                results[method] = {
                    "metric_type": "similarity",
                    "top_matches": top_matches,
                    "time_taken": round(elapsed_time, 4)
                }
            else:
                results[method] = {
                    "metric_type": "distance",
                    "top_matches": top_matches,
                    "time_taken": round(elapsed_time, 4)
                }
        
        return {
            "input": word,
            "input_length": len(word),
            "comparison_mode": "search",
            "dataset_size": len(words_dataset) if words_dataset else 0,
            "algorithm_results": results
        }

@app.get("/dna-lcs/")
def api_find_lcs(seq1: str, seq2: str):
    if not seq1 or not seq2:
        raise HTTPException(status_code=400, detail="Both DNA sequences must be provided.")

    seq1 = seq1.upper()
    seq2 = seq2.upper()
    
    lcs = find_lcs(seq1, seq2)
    return {"longest_common_subsequence": lcs}

@app.get("/dna-similarity/")
def api_dna_similarity(seq1: str, seq2: str, method: Literal['dynamic', 'optimized', 'jaro_winkler', 'damerau_levenshtein'] = 'dynamic'):

    if not seq1 or not seq2:
       raise HTTPException(status_code=400, detail="Both DNA sequences must be provided.")
    
    valid_bases = set('ATGC')
    seq1 = seq1.upper()
    seq2 = seq2.upper()
    
    start_time = time.time()
    
    result = {}
    max_length = max(len(seq1), len(seq2))
    
    if method == 'jaro_winkler':
        similarity = jaro_winkler(seq1, seq2)
        result = {
            "similarity": round(similarity, 4),
            "similarity_percentage": round(similarity * 100, 2),
            "metric_type": "similarity"
        }
    else:
        if method == 'dynamic':
            distance = lev_dynamic(seq1, seq2)
        elif method == 'optimized':
            distance = lev_optimized(seq1, seq2)
        elif method == 'damerau_levenshtein':
            distance = damerau_levenshtein(seq1, seq2)
            
        similarity_percentage = ((max_length - distance) / max_length) * 100
        result = {
            "distance": int(distance),
            "similarity_percentage": round(similarity_percentage, 2),
            "metric_type": "distance"
        }
        
    elapsed_time = time.time() - start_time
    
    return {
        "sequence1_length": len(seq1),
        "sequence2_length": len(seq2),
        "method": method,
        "result": result,
        "time_taken": round(elapsed_time, 4)
    }

@app.get("/dna-full-comparison/")
def api_dna_full_comparison(seq1: str, seq2: str):
    if not seq1 or not seq2:
        raise HTTPException(status_code=400, detail="Both DNA sequences must be provided.")
    
    valid_bases = set('ATGC')
    seq1 = seq1.upper()
    seq2 = seq2.upper()
    
    if not all(base in valid_bases for base in seq1) or not all(base in valid_bases for base in seq2):
        raise HTTPException(status_code=400, detail="DNA sequences should only contain A, T, G, C bases.")
    
    methods = ['dynamic', 'optimized', 'jaro_winkler', 'damerau_levenshtein']
    max_length = max(len(seq1), len(seq2))
    
    results = {}
    for method in methods:
        start_time = time.time()
        
        if method == 'jaro_winkler':
            similarity = jaro_winkler(seq1, seq2)
            results[method] = {
                "similarity": round(similarity, 4),
                "similarity_percentage": round(similarity * 100, 2),
                "metric_type": "similarity",
                "time_taken": round(time.time() - start_time, 4)
            }
        else:
            if method == 'dynamic':
                distance = lev_dynamic(seq1, seq2)
            elif method == 'optimized':
                distance = lev_optimized(seq1, seq2)
            elif method == 'damerau_levenshtein':
                distance = damerau_levenshtein(seq1, seq2)
                
            similarity_percentage = ((max_length - distance) / max_length) * 100
            results[method] = {
                "distance": int(distance),
                "similarity_percentage": round(similarity_percentage, 2),
                "metric_type": "distance",
                "time_taken": round(time.time() - start_time, 4)
            }
    
    start_time = time.time()
    lcs = find_lcs(seq1, seq2)
    lcs_time = round(time.time() - start_time, 4)
    
    return {
        "sequence1": seq1,
        "sequence2": seq2,
        "sequence1_length": len(seq1),
        "sequence2_length": len(seq2),
        "algorithm_results": results,
        "longest_common_subsequence": {
            "sequence": lcs,
            "length": len(lcs),
            "percentage_of_seq1": round((len(lcs) / len(seq1)) * 100, 2) if len(seq1) > 0 else 0,
            "percentage_of_seq2": round((len(lcs) / len(seq2)) * 100, 2) if len(seq2) > 0 else 0,
            "time_taken": lcs_time
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)