import os
from decontaminate import build_ngram_lookup, find_contaminated_questions, find_self_contaminated_questions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def self_sentence_embedding_similarity(train_questions, ds, threshold=0.8, model_name="all-MiniLM-L6-v2", cache_file=".cache/embed_sim_matrix.npy"):
    model = SentenceTransformer(model_name)
    if os.path.exists(cache_file):
        print("Loading embedding similarity matrix from '{}'.".format(cache_file))
        cosine_scores = np.load(cache_file)
    else:
        print("Computing embedding similarity matrix...")
        embeddings = model.encode(train_questions)
        cosine_scores = util.cos_sim(embeddings, embeddings).cpu().numpy()
        cosine_scores = cosine_scores
        np.save(cache_file, cosine_scores)
        print(f"Saved embedding similarity matrix to {cache_file}")
    
    test_questions_id =[0]
    to_remove = set()
    similar_pairs = []
    print("Computing self similarity...", flush=True)
    for j in tqdm(range(1, len(train_questions)), desc="self_filter_TFIDF_similar_strings"):  
        accepted_indices = np.array(test_questions_id)
        similarities = cosine_scores[accepted_indices, j]
        if np.any(similarities >= threshold):
            similar_idx = accepted_indices[np.argmax(similarities >= threshold)]
            to_remove.add(j)
            similar_pairs.append((train_questions[similar_idx], train_questions[j]))
        else:
            test_questions_id.append(j)
            
    not_contaminated_ids = set(range(len(train_questions))) - to_remove
    ds = ds.select(list(not_contaminated_ids))
    
    # Debugging
    print("\nExample remove questions:")
    for q in similar_pairs[:10]:
        print(f"  - {q}")
    
    print(f"\nSelf Filter Sentence Embedding Similar Strings Results:")
    print(f"Total train questions: {len(train_questions)}")
    print(f"Similar questions: {len(to_remove)}")
    print(f"Similarity rate: {(len(to_remove)/len(train_questions)*100):.2f}%")
    print(f"Clean examples remaining: {len(ds)}")
    return ds

def sentence_embedding_similarity(train_questions, test_questions, ds, threshold=0.8, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    print("Computing embedding similarity matrix...")
    embeddings_train = model.encode(train_questions)
    embeddings_test = model.encode(test_questions)
    
    cosine_scores = util.cos_sim(embeddings_test, embeddings_train).cpu().numpy()
    print("Computing similarity...", flush=True)

    to_remove = set()
    similar_pairs = []
    mask = np.any(cosine_scores >= threshold, axis=0)
    for j, has_sim in enumerate(tqdm(mask, desc="sentence_embedding_similarity")):
        if has_sim:
            i = np.argmax(cosine_scores[:, j] >= threshold)
            to_remove.add(j)
            similar_pairs.append((test_questions[i], train_questions[j]))
            
    not_contaminated_ids = set(range(len(train_questions))) - to_remove
    ds = ds.select(list(not_contaminated_ids))
    
    # Debugging
    print("\nExample remove questions:")
    for q in similar_pairs[:10]:
        print(f"  - {q}")
    
    print(f"\nFilter Sentence Embedding Similar Strings Results:")
    print(f"Total train questions: {len(train_questions)}")
    print(f"Similar questions: {len(to_remove)}")
    print(f"Similarity rate: {(len(to_remove)/len(train_questions)*100):.2f}%")
    print(f"Clean examples remaining: {len(ds)}")
    return ds

def filter_TFIDF_similar_strings(train_questions, test_questions, ds, threshold=0.7):
    combined_texts = test_questions + train_questions
    vectorizer = TfidfVectorizer().fit_transform(combined_texts)
    sim_matrix = cosine_similarity(vectorizer[:len(test_questions)], vectorizer[len(test_questions):])

    to_remove = set()
    similar_pairs = []
    
    to_remove = set()
    similar_pairs = []
    scores = sim_matrix[:len(test_questions), :]
    mask = np.any(scores >= threshold, axis=0)
    for j, has_sim in enumerate(tqdm(mask, desc="filter_TFIDF_similar_strings")):
        if has_sim:
            i = np.argmax(scores[:, j] >= threshold)
            to_remove.add(j)
            similar_pairs.append((test_questions[i], train_questions[j]))
            
    not_contaminated_ids = set(range(len(train_questions))) - to_remove
    ds = ds.select(list(not_contaminated_ids))
    
    # Debugging
    print("\nExample remove questions:")
    for q in similar_pairs[:10]:
        print(f"  - {q}")
    
    print(f"\nFilter TF-IDF Similar Strings Results:")
    print(f"Total train questions: {len(train_questions)}")
    print(f"Similar questions: {len(to_remove)}")
    print(f"Similarity rate: {(len(to_remove)/len(train_questions)*100):.2f}%")
    print(f"Clean examples remaining: {len(ds)}")
    return ds

def self_filter_TFIDF_similar_strings(train_questions, ds, threshold=0.7, cache_file=".cache/sim_matrix.npy"):
    if os.path.exists(cache_file):
        sim_matrix = np.load(cache_file)
        print("Loaded cached TFIDF similarity matrix.")
    else:
        print("Computing TFIDF similarity matrix...")
        vectorizer = TfidfVectorizer().fit_transform(train_questions)
        sim_matrix = cosine_similarity(vectorizer)
        np.save(cache_file, sim_matrix)
        print(f"Saved TFIDF similarity matrix to {cache_file}")
    
    test_questions_id =[0]
    to_remove = set()
    similar_pairs = []
    
    for j in tqdm(range(1, len(train_questions)), desc="self_filter_TFIDF_similar_strings"):  
        accepted_indices = np.array(test_questions_id)
        similarities = sim_matrix[accepted_indices, j]
        if np.any(similarities >= threshold):
            similar_idx = accepted_indices[np.argmax(similarities >= threshold)]
            to_remove.add(j)
            similar_pairs.append((train_questions[similar_idx], train_questions[j]))
        else:
            test_questions_id.append(j)
    
    not_contaminated_ids = set(range(len(train_questions))) - to_remove
    ds = ds.select(list(not_contaminated_ids))
    
    # Debugging
    print("\nExample remove questions:")
    for q in similar_pairs[:10]:
        print(f"  - {q}")
    
    print(f"\nSelf Filter TF-IDF Similar Strings Results:")
    print(f"Total train questions: {len(train_questions)}")
    print(f"Similar questions: {len(to_remove)}")
    print(f"Similarity rate: {(len(to_remove)/len(train_questions)*100):.2f}%")
    print(f"Clean examples remaining: {len(ds)}")
    return ds

def file_exists(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)

def decontaminate_train_data(train_questions, test_questions, ds, ngram_size=8):    
    # Build ngram lookups
    train_lookup = build_ngram_lookup(train_questions, ngram_size)
    test_lookup = build_ngram_lookup(test_questions, ngram_size)

    # Find contaminated questions
    contaminated_ids = find_contaminated_questions(train_lookup, test_lookup)

    # Remove contaminated examples
    not_contaminated_ids = set(range(len(train_questions))) - contaminated_ids
    ds = ds.select(list(not_contaminated_ids))
    print(f"\nDecontamination Results:")
    print(f"Total train questions: {len(train_questions)}")
    print(f"Contaminated questions: {len(contaminated_ids)}")
    print(f"Contamination rate: {(len(contaminated_ids)/len(train_questions)*100):.2f}%")
    print(f"Clean examples remaining: {len(ds)}")
    return ds

def self_decontaminate_train_data(train_questions, ds, ngram_size=8):    
    # Build ngram lookups
    train_lookup = build_ngram_lookup(train_questions, ngram_size)
    # Find contaminated questions
    contaminated_ids = find_self_contaminated_questions(train_lookup)

    # Remove contaminated examples
    not_contaminated_ids = set(range(len(train_questions))) - contaminated_ids
    ds = ds.select(list(not_contaminated_ids))
    print(f"\nSelf Decontamination Results:")
    print(f"Total train questions: {len(train_questions)}")
    print(f"Contaminated questions: {len(contaminated_ids)}")
    print(f"Contamination rate: {(len(contaminated_ids)/len(train_questions)*100):.2f}%")
    print(f"Clean examples remaining: {len(ds)}")
    return ds

