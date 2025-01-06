import torch
class EarlyExitWrapper(torch.nn.Module):
    def __init__(self, model, target_layer):
        super().__init__()
        self.model = model
        self.target_layer = target_layer

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Ensure `output_hidden_states=True` is passed
        kwargs["output_hidden_states"] = True

        # Call the base model directly with required arguments
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        hidden_states = outputs.hidden_states
        
        if self.target_layer < 0 or self.target_layer >= len(hidden_states):
            raise ValueError(f"Invalid layer number. Must be between 0 and {len(hidden_states) - 1}")

        # Use the specified layer's hidden states
        early_exit_hidden_state = hidden_states[self.target_layer]
        early_exit_logits = self.model.lm_head(early_exit_hidden_state)

        # Replace the final logits with early exit logits
        outputs.logits = early_exit_logits
        return outputs

    def generate(self, *args, **kwargs):
        # Temporarily patch the model's forward method
        original_forward = self.model.forward
        self.model.forward = self.forward  # Use the wrapper's forward

        try:
            # Use the model's generate method
            return self.model.generate(*args, **kwargs)
        finally:
            # Restore the original forward method
            self.model.forward = original_forward