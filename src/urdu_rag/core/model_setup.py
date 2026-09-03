import logging
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from urdu_rag.configs.schema import LoRAConfig

logger = logging.getLogger(__name__)

class LoRAModelInitializer:
    """Handles model loading, embedding resizing, and LoRA injection."""
    
    def __init__(self, config: LoRAConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self, tokenizer: PreTrainedTokenizerFast) -> PeftModel:
        logger.info(f"Loading base model: {self.config.base_model_name}")
        
        # Note: For the 7B/14B models later, change torch.float32 to torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name, 
            torch_dtype=torch.float32 
        )
        
        original_vocab = base_model.config.vocab_size
        base_model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized embeddings: {original_vocab} -> {len(tokenizer)}")

        logger.info("Injecting LoRA adapters...")
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        peft_model = get_peft_model(base_model, lora_config)
        
        # Log the exact parameter efficiency
        trainable_params, all_params = 0, 0
        for _, param in peft_model.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params
                
        pct = 100 * trainable_params / all_params if all_params > 0 else 0
        logger.info(f"✅ LoRA injected. Trainable: {trainable_params:,} | All: {all_params:,} | %: {pct:.4f}%")
        
        # Save the initialized PEFT model config
        peft_model.save_pretrained(str(self.config.output_dir))
        logger.info(f"Saved LoRA model configuration to {self.config.output_dir}")
        
        return peft_model