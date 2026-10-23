import logging
import torch
from pathlib import Path
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

class UrduRAGGenerator:
    """
    The 'G' in RAG. Takes retrieved context and a query, formats a grounded prompt,
    and generates an answer using the LoRA-adapted LLM.
    """
    
    def __init__(self, model_dir: Path, tokenizer_dir: Path, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        logger.info(f"Loading RAG Generator from {model_dir} on {device}...")
        
        # Load the custom Urdu tokenizer trained in Phase 2
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
        
        # Load the base model (with LoRA adapters if they were merged/saved in the dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            self.model.to(self.device)
            
        self.model.eval()
        logger.info("✅ RAG Generator loaded successfully.")

    def _build_prompt(self, query: str, contexts: List[Tuple[str, float]]) -> str:
        """
        Formats the retrieved documents and query into a strict, grounded Urdu prompt.
        """
        prompt = "آپ ایک ماہر اردو اسسٹنٹ ہیں۔ درج ذیل حوالہ جات کی بنیاد پر سوال کا جواب دیں۔ اگر جواب حوالہ جات میں موجود نہیں ہے، تو واضح طور پر بتا دیں کہ معلومات دستیاب نہیں ہے۔\n\n"
        
        for i, (doc, score) in enumerate(contexts, 1):
            prompt += f"حوالہ {i}: {doc}\n"
            
        prompt += f"\nسوال: {query}\nجواب:"
        return prompt

    def generate_answer(self, query: str, contexts: List[Tuple[str, float]], max_new_tokens: int = 256) -> str:
        """
        Generates a grounded answer based on the retrieved contexts.
        """
        prompt = self._build_prompt(query, contexts)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3, # Low temperature for factual RAG generation
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode only the newly generated tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return answer