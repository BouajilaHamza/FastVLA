from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, Dict, Any

@dataclass
class FastVLAConfig:
    """Configuration class for FastVLA model."""
    
    # Vision Encoder
    vision_encoder_name: str = "google/vit-base-patch16-224"
    image_size: int = 224
    patch_size: int = 16
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    
    # Language Model
    llm_name: str = "meta-llama/Llama-2-7b-hf"
    max_sequence_length: int = 2048
    
    # Action Head
    action_dim: int = 7  # Default for 7-DoF robot arm + gripper
    action_hidden_dim: int = 256
    
    # Training
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    
    # PEFT
    use_peft: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
