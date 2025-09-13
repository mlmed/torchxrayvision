import sys, os
import pytest
import torch
import numpy as np
import torchxrayvision as xrv

def test_mira_sex_model_comprehensive():
    """Comprehensive test for MIRA sex model including interface verification"""
    
    # Test model loading without weights (for testing purposes)
    model = xrv.baseline_models.mira.SexModel(weights=False)
    
    # Test targets
    assert hasattr(model, 'targets'), 'Model should have targets attribute'
    assert model.targets == ["Male", "Female"], 'Targets should be ["Male", "Female"]'
    assert len(model.targets) == 2, 'Should have exactly 2 targets'
    
    # Test model architecture
    assert isinstance(model.model, torch.nn.Module), 'Model should contain a PyTorch module'
    
    # Test forward pass with different input sizes
    test_sizes = [(1, 1, 224, 224), (2, 1, 320, 320), (1, 1, 512, 512)]
    
    for batch_size, channels, height, width in test_sizes:
        img = torch.randn(batch_size, channels, height, width)
        img.requires_grad = True
        
        # Forward pass
        with torch.no_grad():
            outputs = model(img)
        
        # Check output shape
        assert outputs.shape == (batch_size, 2), f'Output shape should be ({batch_size}, 2) but got {outputs.shape}'
        
        # Test softmax conversion
        with torch.no_grad():
            probs = torch.softmax(outputs, 1)
        
        # Check probabilities sum to 1
        prob_sums = torch.sum(probs, dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), 'Probabilities should sum to 1'
        
        # Test gradient computation (need to compute outputs with grad enabled)
        outputs_with_grad = model(img)
        pred = outputs_with_grad[:, model.targets.index("Male")]
        grads = torch.autograd.grad(pred.sum(), img)[0]
        assert grads.shape == img.shape, 'Gradients should have same shape as input'
        assert not torch.isnan(grads).any(), 'Gradients should not contain NaN values'
    
    # Test the expected interface
    img = torch.randn(1, 1, 224, 224)
    
    # Test the exact interface specified in the requirements
    model = xrv.baseline_models.mira.SexModel(weights=False)
    assert model.targets == ["Male", "Female"], 'targets should return ["Male", "Female"]'
    
    with torch.no_grad():
        outputs = torch.softmax(model(img), 1)
    
    prediction_dict = dict(zip(model.targets, outputs.tolist()[0]))
    
    # Verify prediction dict structure
    assert isinstance(prediction_dict, dict), 'Should return a dictionary'
    assert set(prediction_dict.keys()) == {"Female", "Male"}, 'Dictionary should have Female and Male keys'
    assert all(isinstance(v, float) for v in prediction_dict.values()), 'All values should be floats'
    assert all(0 <= v <= 1 for v in prediction_dict.values()), 'All probabilities should be between 0 and 1'
    assert abs(sum(prediction_dict.values()) - 1.0) < 1e-6, 'Probabilities should sum to 1'
    
    print("All tests passed! The interface works as expected.")
    print(f"Example prediction: {prediction_dict}")

if __name__ == "__main__":
    test_mira_sex_model_comprehensive()
