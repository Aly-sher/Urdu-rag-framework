import lang2vec.lang2vec as l2v
import numpy as np
from scipy.spatial.distance import euclidean

def calculate_typological_distances():
    """
    Step 1: Typology-Driven Language Alignment
    Extracts syntactic and phonological vectors from URIEL+ and calculates 
    the geometric distance between Urdu and potential anchor languages.
    """
    
    # ISO 639-3 language codes
    target_lang = 'urd'  # Urdu
    anchor_langs = {
        'hin': 'Hindi',     # Typologically similar (Indic, SOV)
        'fas': 'Persian',   # Lexically similar (Perso-Arabic script, heavy loanwords)
        'eng': 'English'    # Baseline/Default (SVO, Latin script)
    }

    # URIEL+ feature sets to extract
    feature_sets = ['syntax_average', 'phonology_average']
    
    print("--- Extracting URIEL+ Typological Vectors ---")
    print("(Note: It may download a small dataset on the first run)\n")
    
    vectors = {}
    for lang_code, lang_name in {target_lang: 'Urdu', **anchor_langs}.items():
        vectors[lang_name] = {}
        for feature in feature_sets:
            # Query one feature at a time to get a clean result
            res = l2v.get_features([lang_code], [feature])
            
            # The result is a dict: {lang_code: [list of values]}
            raw_vec = res[lang_code]
            
            # Clean the data: replace '--' (missing) with 0.0, and convert to float
            clean_vec = [0.0 if str(v) == '--' else float(v) for v in raw_vec]
            vec = np.array(clean_vec, dtype=np.float32)
            
            vectors[lang_name][feature] = vec
            print(f"Extracted {feature} for {lang_name} | Vector shape: {vec.shape}")

    print("\n--- Calculating Geometric Distances (Procrustes Prep) ---")
    
    results = {}
    
    for anchor_code, anchor_name in anchor_langs.items():
        distances = {}
        for feature in feature_sets:
            urdu_vec = vectors['Urdu'][feature]
            anchor_vec = vectors[anchor_name][feature]
            
            # Calculate Euclidean distance (L2 norm)
            dist = euclidean(urdu_vec, anchor_vec)
            distances[feature] = dist
            
        results[anchor_name] = distances

    # Print the results clearly
    print("\n" + "="*60)
    print("TYPOLOGICAL DISTANCE RESULTS (Lower distance = closer linguistic relative)")
    print("="*60)
    for anchor_name, dists in results.items():
        print(f"\nAnchor: {anchor_name}")
        for feature, dist in dists.items():
            print(f"  - {feature.capitalize()} Distance: {dist:.4f}")
            
    # Determine the best overall anchor
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    
    # Calculate average distance across features for each anchor
    avg_distances = {}
    for anchor_name, dists in results.items():
        avg_distances[anchor_name] = np.mean(list(dists.values()))
        
    best_anchor = min(avg_distances, key=avg_distances.get)
    print(f"✅ Based on URIEL+ typological features, the closest anchor language to Urdu is: {best_anchor.upper()}")
    print(f"💡 This language will be used for the Procrustes Alignment matrix (W) in the next step.")

if __name__ == "__main__":
    calculate_typological_distances()