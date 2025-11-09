import torch
import torch.nn as nn
from typing import Optional
from transformers.modeling_utils import PreTrainedModel
from unsloth import (
    FastLanguageModel,
    FastVisionModel,
    patch_forward,
    patch_model,
    patch_peft_model,
    patch_saving_functions,
    patch_tokenizer,
)
from .config import FastVLAConfig
from .kernels import (
    vision_language_fusion_forward,
    multi_cam_pack_forward,
)
from .optimization import (
    get_quantization_config,
    enable_gradient_checkpointing,
    get_peft_config,
)

class FastVLAModel(PreTrainedModel):
    """
    FastVLA: Efficient Vision-Language-Action model for robotics.
    Combines a vision encoder, language model, and action head.
    """
    config_class = FastVLAConfig
    
    def __init__(self, config: FastVLAConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize vision encoder with Unsloth optimizations
        self.vision_encoder = FastVisionModel.from_pretrained(
            config.vision_encoder_name,
            load_in_4bit=config.load_in_4bit,
            token = None,  # Add your HF token if needed
            device_map = "auto",
            torch_dtype = torch.bfloat16,
            attn_implementation = "flash_attention_2",
        )
        
        # Initialize language model with Unsloth optimizations
        self.llm, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = config.llm_name,
            max_seq_length = config.max_sequence_length,
            dtype = torch.bfloat16,
            load_in_4bit = config.load_in_4bit,
            token = None,  # Add your HF token if needed
            device_map = "auto",
            rope_scaling = {"type": "dynamic", "factor": 2.0},
            attn_implementation = "flash_attention_2",
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.vision_encoder.gradient_checkpointing_enable()
        self.llm.gradient_checkpointing_enable()
        
        # Action head with Unsloth optimizations
        # Use LoRA for parameter-efficient fine-tuning
        if config.use_peft:
            from peft import LoraConfig
            peft_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm = FastLanguageModel.get_peft_model(self.llm, peft_config)
        
        # Initialize action head with Kaiming initialization
        self.action_head = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, config.action_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.action_hidden_dim, config.action_dim),
            nn.Tanh()
        )
        nn.init.kaiming_normal_(self.action_head[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.action_head[2].weight, mode='fan_in', nonlinearity='linear')
        
        # Project vision embeddings to LLM hidden size with initialization
        self.vision_proj = nn.Linear(
            self.vision_encoder.config.hidden_size,
            self.llm.config.hidden_size
        )
        nn.init.xavier_uniform_(self.vision_proj.weight)
        
        # Apply Unsloth patches for optimization
        self.llm = patch_model(self.llm)
        patch_saving_functions()
        patch_forward(self.llm)
        
        # Enable flash attention and other optimizations
        self.llm = self.llm.to_bettertransformer()
        self.vision_encoder = self.vision_encoder.to_bettertransformer()
    
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the model.
        
        Args:
            pixel_values: (batch_size, num_cameras, channels, height, width)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, action_dim)
            
        Returns:
            action_preds: (batch_size, action_dim)
            loss: Optional[torch.Tensor]
        """
        batch_size = pixel_values.size(0)
        num_cameras = pixel_values.size(1)
        
        # Process vision encoder for each camera view
        # Use efficient batched processing when possible
        visual_features = []
        for cam_idx in range(num_cameras):
            # Process each camera view through vision encoder
            cam_images = pixel_values[:, cam_idx]  # (batch_size, C, H, W)
            vision_outputs = self.vision_encoder(
                pixel_values=cam_images,
                return_dict=True
            )
            # Get visual tokens and project to LLM hidden size
            cam_features = self.vision_proj(vision_outputs.last_hidden_state)  # (batch_size, seq_len, hidden_size)
            visual_features.append(cam_features)
        
        # Average visual features across camera views
        visual_features = torch.stack(visual_features, dim=0).mean(dim=0)  # (batch_size, seq_len, hidden_size)
        
        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (batch_size, seq_len, hidden_size)
        
        # Fuse visual and text features using custom kernel
        # This is more efficient than cross-attention for simple fusion
        if visual_features.size(1) != text_embeds.size(1):
            # If sequence lengths don't match, pool visual features
            visual_features = visual_features.mean(dim=1, keepdim=True)  # (batch_size, 1, hidden_size)
            visual_features = visual_features.expand(-1, text_embeds.size(1), -1)  # (batch_size, seq_len, hidden_size)
        
        # Use custom fusion kernel for efficient vision-language fusion
        fused_embeds = vision_language_fusion_forward(visual_features, text_embeds)
        
        # Forward pass through LLM with Unsloth optimizations
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            outputs = self.llm(
                inputs_embeds=fused_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,  # Disable KV cache for training
            )
        
        # Get the last hidden state for action prediction
        last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        
        # Use mean pooling for action prediction
        pooled_output = last_hidden_state.mean(dim=1)  # (batch_size, hidden_size)
        action_preds = self.action_head(pooled_output)  # (batch_size, action_dim)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = nn.MSELoss()(action_preds, labels)
            
        return action_preds, loss
    
    def generate(self, images, input_ids, **kwargs):
        """Generate actions from images and text."""
        # Get model predictions
        action_preds, _ = self(images, input_ids)
        return action_preds
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: Optional[str] = None,
        vision_encoder_name: Optional[str] = None,
        llm_name: Optional[str] = None,
        config: Optional[FastVLAConfig] = None,
        load_in_4bit: bool = True,
        max_seq_length: int = 2048,
        gradient_checkpointing: bool = True,
        use_peft: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        token: Optional[str] = None,
        device_map: str = "auto",
        **kwargs
    ):
        """
        Load a FastVLA model from pretrained components.
        
        Args:
            model_name: Name of the full model (if available)
            vision_encoder_name: Name of the vision encoder model
            llm_name: Name of the language model
            config: FastVLAConfig object (optional)
            load_in_4bit: Whether to load in 4-bit quantization
            max_seq_length: Maximum sequence length
            gradient_checkpointing: Whether to enable gradient checkpointing
            use_peft: Whether to use PEFT (LoRA)
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            token: HuggingFace token for private models
            device_map: Device mapping strategy
            **kwargs: Additional arguments
            
        Returns:
            FastVLAModel instance
        """
        # Create config if not provided
        if config is None:
            config = FastVLAConfig(
                vision_encoder_name=vision_encoder_name or "google/vit-base-patch16-224",
                llm_name=llm_name or "meta-llama/Llama-2-7b-hf",
                max_sequence_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                use_peft=use_peft,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        
        # Create model instance
        model = cls(config)
        
        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            enable_gradient_checkpointing(model)
        
        return model
