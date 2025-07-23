import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization
    
    More stable than LayerNorm and doesn't require computing mean
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate root mean square
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE)
    
    Encodes position information through rotation of query/key vectors
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Compute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute rotation matrices
        self._precompute_rotations()
    
    def _precompute_rotations(self):
        pos = torch.arange(self.max_seq_len)
        freqs = torch.einsum('i,j->ij', pos, self.inv_freq)
        
        # Create rotation matrices
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the dimensions"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position encoding to query and key tensors"""
        # q, k have shape [batch_size, num_heads, seq_len, head_dim]
        batch_size, num_heads, _, head_dim = q.shape
        
        # Get relevant cos/sin values for the sequence length
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        # cos/sin have shape [1, 1, seq_len, dim//2] but we need [batch, heads, seq_len, head_dim]
        # We need to repeat the cos/sin values to match the full head dimension
        cos = cos.repeat(1, 1, 1, 2)  # Now [1, 1, seq_len, dim]
        sin = sin.repeat(1, 1, 1, 2)  # Now [1, 1, seq_len, dim]
        
        # Only use the part that matches head_dim
        cos = cos[..., :head_dim]
        sin = sin[..., :head_dim]
        
        # Apply rotation
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rot, k_rot


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) variant
    
    Splits input into two parts and uses one to gate the other
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # Project to 2x hidden dimension (for gating)
        self.w_gate = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, input_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate and up projection
        gate = F.silu(self.w_gate(x))  # SwiGLU activation
        up = self.w_up(x)
        
        # Element-wise multiplication and down projection
        return self.w_down(gate * up)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with RoPE support"""
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Q, K, V projections (no bias as per paper)
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)
        
        # Rotary positional encoding
        self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position encoding
        q, k = self.rope(q, k, seq_len)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.w_o(out)
        
        return out


class TransformerBlock(nn.Module):
    """Transformer encoder block with modern enhancements
    
    Includes:
    - RMSNorm instead of LayerNorm
    - Rotary positional encoding
    - Gated Linear Units
    - No bias terms
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        
        # Attention block
        self.attention_norm = RMSNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads, max_seq_len, dropout)
        
        # Feed-forward block
        self.ffn_norm = RMSNorm(dim)
        self.ffn = GatedLinearUnit(dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Attention block with residual connection
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward block with residual connection
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x


def truncated_lecun_normal_(tensor: torch.Tensor, truncation: float = 2.0):
    """Initialize tensor with truncated LeCun normal distribution"""
    fan_in = tensor.shape[1] if len(tensor.shape) >= 2 else tensor.shape[0]
    std = math.sqrt(1.0 / fan_in)
    
    with torch.no_grad():
        # Generate normal distribution and truncate
        tensor.normal_(0, std)
        tensor.clamp_(-truncation * std, truncation * std)