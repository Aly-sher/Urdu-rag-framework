import logging
import re
from typing import List, Set

logger = logging.getLogger(__name__)

class UrduDataProcessor:
    """
    Handles corpus quality filtering and strict train/eval decontamination.
    Uses pure Python Unicode heuristics for instant, dependency-free language filtering.
    """
    
    def __init__(self):
        # Urdu uses the Arabic Unicode block (0600-06FF) and Arabic Supplement (0750-077F)
        self.urdu_char_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F]')

    def filter_by_urdu_heuristic(self, texts: List[str], threshold: float = 0.6) -> List[str]:
        """
        Filters out non-Urdu or heavily code-switched text by checking the 
        ratio of Urdu Unicode characters to total non-whitespace characters.
        """
        logger.info(f"Filtering {len(texts)} documents using pure Python Unicode heuristic...")
        clean_texts = []
        dropped_count = 0
        
        for text in texts:
            # Remove whitespace for accurate character counting
            text_no_space = re.sub(r'\s+', '', text)
            if not text_no_space:
                dropped_count += 1
                continue
                
            # Count Urdu characters
            urdu_chars = len(self.urdu_char_pattern.findall(text_no_space))
            urdu_ratio = urdu_chars / len(text_no_space)
            
            # If >60% of the characters are Urdu script, keep it
            if urdu_ratio >= threshold:
                clean_texts.append(text)
            else:
                dropped_count += 1
                
        logger.info(f"✅ Retained {len(clean_texts)}/{len(texts)} documents. Dropped {dropped_count} (non-Urdu/code-switched).")
        return clean_texts

    def decontaminate(self, train_texts: List[str], eval_texts: List[str], ngram_size: int = 5) -> List[str]:
        """
        Removes training documents that have high n-gram overlap with evaluation sets.
        Prevents data leakage. (Standard 5 to 13-gram is used in standard LLM evals).
        """
        logger.info(f"Decontaminating training corpus against {len(eval_texts)} eval documents (n-gram={ngram_size})...")
        
        # Extract n-grams from eval set
        eval_ngrams: Set[str] = set()
        for text in eval_texts:
            tokens = text.split()
            for i in range(len(tokens) - ngram_size + 1):
                ngram = " ".join(tokens[i:i+ngram_size])
                eval_ngrams.add(ngram)
                
        logger.info(f"Extracted {len(eval_ngrams)} unique {ngram_size}-grams from eval set.")
        
        # Filter train set
        clean_train = []
        contaminated_count = 0
        
        for text in train_texts:
            tokens = text.split()
            is_contaminated = False
            for i in range(len(tokens) - ngram_size + 1):
                ngram = " ".join(tokens[i:i+ngram_size])
                if ngram in eval_ngrams:
                    is_contaminated = True
                    break
            
            if not is_contaminated:
                clean_train.append(text)
            else:
                contaminated_count += 1
                
        logger.info(f"✅ Removed {contaminated_count} contaminated documents. Retained {len(clean_train)}.")
        return clean_train