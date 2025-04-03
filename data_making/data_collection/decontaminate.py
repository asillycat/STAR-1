import collections
from tqdm import tqdm

def normalize_string(text):
    """Basic string normalization."""
    # Convert to lowercase and normalize whitespace
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    return text

def word_ngrams(text, n):
    """Generate word-level n-grams from text."""
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

def build_ngram_lookup(documents, ngram_size=13):
    """Build ngram lookup for documents."""
    print(f"Building {ngram_size}-gram lookup...")
    lookup = collections.defaultdict(set)

    for doc_id, document in enumerate(tqdm(documents)):
        normalized_text = normalize_string(document)
        ngrams = word_ngrams(normalized_text, ngram_size)
        for ngram in ngrams:
            lookup[ngram].add(doc_id)
    
    return lookup


def find_contaminated_questions(train_lookup, test_lookup):
    """Find overlapping documents based on ngram matches."""
    contaminated_ids = set()
    matched_ngrams = []  # For debugging
    
    for ngram, train_doc_ids in tqdm(train_lookup.items(), desc="Checking overlaps"):
        if ngram in test_lookup:
            contaminated_ids.update(train_doc_ids)
            matched_ngrams.append(ngram)
    
    # # Print some example matches for inspection
    # if matched_ngrams:
    #     print("\nExample matching n-grams:")
    #     for ngram in matched_ngrams[:5]:  # Show first 5 matches
    #         print(f"  - {ngram}")
    
    return contaminated_ids

def find_self_contaminated_questions(train_lookup):
    """Find overlapping documents based on ngram matches."""
    contaminated_ids = set()
    matched_ngrams = []  # For debugging
    
    for ngram, train_doc_ids in tqdm(train_lookup.items(), desc="Checking overlaps"):
        if len(train_doc_ids) > 1:
            train_doc_ids.pop()
            contaminated_ids.update(train_doc_ids)
            matched_ngrams.append(ngram)
    
    # # Print some example matches for inspection
    # if matched_ngrams:
    #     print("\nExample matching n-grams:")
    #     for ngram in matched_ngrams[:5]:  # Show first 5 matches
    #         print(f"  - {ngram}")
    
    return contaminated_ids