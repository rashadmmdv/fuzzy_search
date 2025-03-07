Install dependencies:
    pip install -r requirements.txt

Start the server:
    uvicorn app:app --reload

Interactive API documentation is available at:
Swagger UI: http://localhost:8000/docs


Search Methods:

/search/?word={word}&method={method}
/search/?word=appl&method=jaro_winkler

/search-full-comparison/?word={word}
/search-full-comparison/?word=appl

DNA Analysis

/dna-lcs/?seq1={sequence1}&seq2={sequence2}
/dna-lcs/?seq1=ACCGGTCGAGTGCGCGGAAGCCGGCCGAA&seq2=GTCGTTCGGAATGCCGTTGCTCTGTAAA

/dna-similarity/?seq1={sequence1}&seq2={sequence2}&method={method}
/dna-similarity/?seq1=ACCGGTCGAGTGCGCGGAAGCCGGCCGAA&seq2=GTCGTTCGGAATGCCGTTGCTCTGTAAA&method=dynamic

/dna-full-comparison/?seq1={sequence1}&seq2={sequence2}
/dna-full-comparison/?seq1=ACCGGTCGAGTGCGCGGAAGCCGGCCGAA&seq2=GTCGTTCGGAATGCCGTTGCTCTGTAAA