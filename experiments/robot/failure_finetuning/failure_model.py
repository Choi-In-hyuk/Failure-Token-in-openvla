"""
Wrapper model that adds failure detection capability to pretrained OpenVLA model.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


class OpenVLAWithFailureDetection(nn.Module):
    def __init__(self, pretrained_checkpoint, device="cuda", load_in_8bit=False, load_in_4bit=False):
        super().__init__()
        self.device = device
        self.pretrained_checkpoint = pretrained_checkpoint
        
        # Register OpenVLA classes to HF
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        
        # Load base OpenVLA model
        print(f"[*] Loading base OpenVLA model from {pretrained_checkpoint}")
        self.openvla_model = AutoModelForVision2Seq.from_pretrained(
            pretrained_checkpoint,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Move to device if not quantized
        if not load_in_8bit and not load_in_4bit:
            self.openvla_model = self.openvla_model.to(device)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            pretrained_checkpoint, 
            trust_remote_code=True
        )
        
        # Get hidden size from language model config
        hidden_size = self.openvla_model.language_model.config.hidden_size  # Should be 4096
        
        # Failure detection head
        self.failure_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # [success_logit, failure_logit]
        )
        
        # Move failure head to device and convert to bfloat16
        self.failure_head = self.failure_head.to(device, dtype=torch.bfloat16)
        
        print(f"[*] Added failure detection head with input size {hidden_size}")
        
    def freeze_base_model(self):
        """Freeze all parameters in the base OpenVLA model"""
        for param in self.openvla_model.parameters():
            param.requires_grad = False
        print("[*] Base OpenVLA model parameters frozen")
        
    def unfreeze_base_model(self):
        """Unfreeze all parameters in the base OpenVLA model"""
        for param in self.openvla_model.parameters():
            param.requires_grad = True
        print("[*] Base OpenVLA model parameters unfrozen")
        
    def get_trainable_parameters(self):
        """Get only the trainable parameters (failure head)"""
        return self.failure_head.parameters()
    
    def forward(self, inputs, return_hidden_states=True):
        """
        Forward pass that returns both action predictions and failure detection
        
        Args:
            inputs: Processed inputs from the processor
            return_hidden_states: Whether to return hidden states for failure detection
            
        Returns:
            dict containing:
                - action_logits: Action predictions from base model
                - failure_logits: Failure detection logits [success, failure]
                - hidden_states: Language model hidden states (if return_hidden_states=True)
        """
        # Get outputs from base OpenVLA model
        with torch.set_grad_enabled(self.training):
            base_outputs = self.openvla_model(**inputs, output_hidden_states=return_hidden_states)
            
        results = {
            'action_logits': base_outputs.logits if hasattr(base_outputs, 'logits') else None
        }
        
        # Get failure detection if hidden states available
        if return_hidden_states and hasattr(base_outputs, 'hidden_states'):
            # Use last layer, last token hidden state
            last_hidden_state = base_outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_size]
            failure_logits = self.failure_head(last_hidden_state)
            results['failure_logits'] = failure_logits
            results['hidden_states'] = last_hidden_state
            
        return results
    
    def predict_action_and_failure(self, image, task_description, unnorm_key, center_crop=False):
        """
        Predict both action and failure detection for a single step
        
        Args:
            image: PIL Image
            task_description: Task instruction string
            unnorm_key: Normalization key for action denormalization
            center_crop: Whether to apply center crop
            
        Returns:
            dict containing:
                - action: Predicted action array
                - failure_prob: Failure probability [0, 1]
                - failure_logits: Raw failure logits
        """
        self.eval()
        
        # Prepare prompt (same as original OpenVLA)
        if "openvla-v01" in self.pretrained_checkpoint:
            prompt = (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                f"USER: What action should the robot take to {task_description.lower()}? ASSISTANT:"
            )
        else:
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        
        # Process inputs
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        
        with torch.no_grad():
            # Get action prediction using original method
            action = self.openvla_model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            
            # Get failure detection
            outputs = self.forward(inputs, return_hidden_states=True)
            failure_logits = outputs['failure_logits']
            failure_prob = torch.softmax(failure_logits, dim=-1)[0, 1].cpu().item()  # Get failure probability
            
        return {
            'action': action,
            'failure_prob': failure_prob,
            'failure_logits': failure_logits.cpu(),
            'is_failure': failure_prob > 0.5
        }
    
    def save_pretrained(self, save_directory):
        """Save the model with both base model and failure head"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the base model
        self.openvla_model.save_pretrained(save_directory)
        
        # Save failure head separately
        torch.save({
            'failure_head_state_dict': self.failure_head.state_dict(),
            'config': {
                'pretrained_checkpoint': self.pretrained_checkpoint,
                'hidden_size': self.openvla_model.language_model.config.hidden_size
            }
        }, os.path.join(save_directory, 'failure_head.pt'))
        
        print(f"[*] Model saved to {save_directory}")
        
    @classmethod
    def from_pretrained(cls, save_directory, device="cuda"):
        """Load the model from saved directory"""
        import os
        import json
        
        # Load failure head config
        failure_head_path = os.path.join(save_directory, 'failure_head.pt')
        checkpoint = torch.load(failure_head_path, map_location=device)
        config = checkpoint['config']
        
        # Create model instance using original checkpoint path for processor
        original_checkpoint = config['pretrained_checkpoint']
        model = cls(
            pretrained_checkpoint=original_checkpoint,  # Use original checkpoint for processor
            device=device
        )
        
        # Load the saved base model instead
        model.openvla_model = AutoModelForVision2Seq.from_pretrained(
            save_directory,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)
        
        # Load failure head weights
        model.failure_head.load_state_dict(checkpoint['failure_head_state_dict'])
        
        print(f"[*] Model loaded from {save_directory}")
        return model