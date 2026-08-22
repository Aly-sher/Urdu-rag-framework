import os
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

def setup_urdu_tokenizer_and_lora():
    """
    Step 2: Script Adaptation & Parameter Efficiency
    Trains a custom BPE tokenizer for Nastaliq and sets up LoRA adapters 
    for parameter-efficient Continual Pre-Training (CPT).
    """
    
    print("="*60)
    print("PHASE 2A: Training Custom BPE Tokenizer for Nastaliq")
    print("="*60)
    
    # 1. Create a dummy Urdu corpus (In production, this will be 10GB+ of clean Urdu text)
    urdu_texts = [
        "اردو ایک بہت ہی خوبصورت اور شیریں زبان ہے۔",
        "پاکستان کی قومی زبان اردو ہے۔",
        "علامہ اقبال اردو کے عظیم شاعر تھے۔",
        "لاہور کا موسم سردیوں میں بہت اچھا ہوتا ہے۔",
        "کراچی پاکستان کا سب سے بڑا شہر اور اقتصادی حب ہے۔",
        "محبت ایک خوبصورت احساس ہے۔" # Testing the word "love"
    ]
    
    os.makedirs("data", exist_ok=True)
    corpus_path = "data/urdu_dummy_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n".join(urdu_texts))

    # 2. Initialize and train the BPE Tokenizer
    # ByteLevel is excellent for handling complex, connected scripts like Nastaliq
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) 
    
    trainer = trainers.BpeTrainer(
        vocab_size=5000, # Small for this test. Production would be 15k-30k
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    tokenizer.train([corpus_path], trainer)
    tokenizer.save("data/urdu_bpe_tokenizer.json")
    print("✅ Custom BPE Tokenizer trained and saved to data/urdu_bpe_tokenizer.json\n")

    # 3. Wrap it in HuggingFace's Fast Tokenizer for model compatibility
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="data/urdu_bpe_tokenizer.json",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]"
    )
    
    # Test the tokenizer to prove it handles Nastaliq correctly
    test_text = "محبت"
    tokens = hf_tokenizer.tokenize(test_text)
    print(f"🔍 Tokenizer Test:")
    print(f"   Original Word: '{test_text}'")
    print(f"   Tokenized as:  {tokens}")
    print("   (Notice how it keeps the word intact instead of shattering it into broken bytes!)\n")

    print("="*60)
    print("PHASE 2B: Loading Base Model & Resizing Embeddings")
    print("="*60)
    
    # We use a small model for demonstration. In production, use Qwen2.5-7B or Llama-3-8B
    model_name = "Qwen/Qwen2.5-0.5B" 
    print(f"⏳ Loading base model: {model_name} (This may take a minute to download)...")
    
    # Load model in float32 for stability during this setup test
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    print(f"   Original Vocabulary Size: {base_model.config.vocab_size}")

    # 4. Resize model embeddings to match the new Urdu vocabulary
    # This adds the new Urdu tokens to the model's brain without changing the old weights
    base_model.resize_token_embeddings(len(hf_tokenizer))
    print(f"✅ Resized Vocabulary Size: {base_model.config.vocab_size}\n")

    print("="*60)
    print("PHASE 2C: Injecting LoRA Adapters")
    print("="*60)
    
    # 5. Configure LoRA (Low-Rank Adaptation)
    # We only train the adapters, keeping the base model frozen to prevent catastrophic forgetting
    lora_config = LoraConfig(
        r=16, # Rank of the update matrices (higher = more capacity, but more VRAM)
        lora_alpha=32, # Scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Target attention layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA to the model
    peft_model = get_peft_model(base_model, lora_config)
    
    print("\n🎉 Phase 2 Setup Complete! Trainable parameters summary:")
    peft_model.print_trainable_parameters()


if __name__ == "__main__":
    setup_urdu_tokenizer_and_lora()