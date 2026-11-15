import pytest
import torch
from dna_serve import DNAServe, create_dna_serve

class TestDNAServe:
    def test_init(self):
        m = DNAServe(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = DNAServe(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_dna_serve()
        assert isinstance(m, DNAServe)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
