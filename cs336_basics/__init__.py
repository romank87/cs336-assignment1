import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .bpe_tokenizer import run_train_bpe