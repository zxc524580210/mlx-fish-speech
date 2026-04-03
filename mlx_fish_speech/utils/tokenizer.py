"""
Tokenizer wrapper for Fish-Speech using tiktoken.
"""

from pathlib import Path
from typing import List, Optional

import tiktoken


class FishTokenizer:
    """
    Tokenizer for Fish-Speech using tiktoken BPE.
    """
    
    # Special tokens
    PAD_TOKEN = "<|pad|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    AUDIO_START = "<|audio|>"
    AUDIO_END = "<|/audio|>"
    
    def __init__(self, tokenizer_path: Optional[str] = None):
        """
        Initialize tokenizer.
        
        Args:
            tokenizer_path: Path to tiktoken tokenizer file
        """
        if tokenizer_path:
            # Load custom tokenizer
            with open(tokenizer_path, "rb") as f:
                self.encoding = tiktoken.Encoding(
                    name="fish_speech",
                    pat_str=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
                    mergeable_ranks=self._load_tiktoken_bpe(f.read()),
                    special_tokens={
                        self.PAD_TOKEN: 0,
                        self.BOS_TOKEN: 1,
                        self.EOS_TOKEN: 2,
                        self.AUDIO_START: 3,
                        self.AUDIO_END: 4,
                    },
                )
        else:
            # Use default cl100k_base as fallback
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
    
    def _load_tiktoken_bpe(self, data: bytes) -> dict:
        """Load BPE ranks from tiktoken file."""
        import base64
        ranks = {}
        for line in data.decode().splitlines():
            if line:
                token, rank = line.split()
                ranks[base64.b64decode(token)] = int(rank)
        return ranks
    
    def encode(
        self, 
        text: str, 
        add_bos: bool = True, 
        add_eos: bool = False
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_bos: Add beginning of sequence token
            add_eos: Add end of sequence token
            
        Returns:
            List of token IDs
        """
        tokens = self.encoding.encode(text, allowed_special="all")
        
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text string
        """
        # Filter special tokens
        tokens = [t for t in tokens if t >= 5]
        return self.encoding.decode(tokens)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.encoding.n_vocab
    
    @classmethod
    def from_pretrained(cls, path: str) -> "FishTokenizer":
        """Load tokenizer from pretrained model directory."""
        tokenizer_path = Path(path) / "tokenizer.tiktoken"
        if tokenizer_path.exists():
            return cls(str(tokenizer_path))
        return cls()
