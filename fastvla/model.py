import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
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
    
    @torch.compile(fullgraph=False, mode='max-autotune')
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the model.
        
        Args:
            images: (batch_size, num_cameras, channels, height, width)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, action_dim)
            
        Returns:
            action_preds: (batch_size, action_dim)
            loss: Optional[torch.Tensor]
        """
        batch_size = pixel_values.size(0)
        num_cameras = pixel_values.size(1)
        
        # Process each camera view with torch.compile optimization
        visual_features = []
        for cam_idx in range(num_cameras):
            # Use torch.compile for the vision encoder
            @torch.compile(fullgraph=False, mode='max-autotune')
            def process_view(x):
                return self.vision_encoder(
                    pixel_values=x,
                    return_dict=True
                ).last_hidden_state
            
            # Process view with optimized function
            cam_outputs = process_view(pixel_values[:, cam_idx])
            visual_features.append(self.vision_proj(cam_outputs))
        
        # Average features across camera views
        visual_features = torch.stack(visual_features).mean(dim=0)  # (batch_size, seq_len, hidden_size)
        
        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Fuse visual and text features with memory-efficient attention
        # Using Unsloth's optimized attention implementation
        visual_features = torch.stack(visual_features).mean(dim=0)  # Average across views
        
        # Cross-attention between text and visual features
        # This is more efficient than simple concatenation
        cross_attention_outputs = self.llm.model.model.decoder.layers[0].encoder_attn(
            hidden_states=text_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=visual_features,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        
        combined_embeds = cross_attention_outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # Forward pass through LLM with Unsloth optimizations
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            outputs = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,  # Disable KV cache for training
            )
        
        # Get the last hidden state for action prediction
        last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        
        # Use the [CLS] token or mean pooling for action prediction
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
