from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

@dataclass
class UnslothVLACollator:
    """
    Data collator for FastVLA that handles:
    - Multi-camera image batching
    - Variable-length sequences
    - Mixed data types (images, states, actions, text)
    """
    
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 512
    padding: Union[bool, str] = True
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Args:
            features: List of feature dictionaries from the dataset
            
        Returns:
            Batch dictionary with padded/stacked tensors
        ""
        batch = {}
        
        # Handle images (multiple camera views)
        if 'images' in features[0]:
            all_images = {}
            # Get all unique camera keys across the batch
            camera_keys = set()
            for feature in features:
                camera_keys.update(feature['images'].keys())
            
            # Stack images for each camera view
            for cam in camera_keys:
                images = [f['images'][cam] for f in features if cam in f['images']]
                if images:
                    all_images[cam] = torch.stack(images)
            
            batch['pixel_values'] = all_images
        
        # Handle states
        if 'states' in features[0]:
            states = [torch.as_tensor(f['states']) for f in features]
            batch['states'] = torch.stack(states)
        
        # Handle actions
        if 'actions' in features[0]:
            actions = [torch.as_tensor(f['actions']) for f in features]
            batch['labels'] = torch.stack(actions)
        
        # Handle text instructions
        if 'instructions' in features[0]:
            texts = [f['instructions'] for f in features]
            text_inputs = self.tokenizer(
                texts,
                padding=self.padding,
                truncation=True,
                max_length=self.max_length,
                return_tensors=self.return_tensors,
                return_attention_mask=True,
                return_token_type_ids=False,
            )
            batch.update({
                'input_ids': text_inputs['input_ids'],
                'attention_mask': text_inputs['attention_mask'],
            })
        
        return batch
