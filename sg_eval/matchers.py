from typing import Dict, List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    import torch
    from transformers import AutoModel, AutoTokenizer, CLIPModel
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


class _TextMatcherBase:
    """Shared helper for text-embedding based ranking."""

    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def precompute(self, texts: List[str]):
        raise NotImplementedError

    def get_embeddings(self, vocab: List[str]) -> np.ndarray:
        return np.stack([self.encode(v) for v in vocab])

    def rank_in_vocab(
        self, pred_label: str, gt_label: str,
        vocab: List[str], vocab_embs: np.ndarray,
    ) -> int:
        pred_emb = self.encode(pred_label)
        sims = vocab_embs @ pred_emb
        gt_idx = vocab.index(gt_label)
        gt_sim = sims[gt_idx]
        return int(np.sum(sims > gt_sim)) + 1


def _coerce_text_features_to_tensor(output, model=None):
    """Return a tensor from Transformers text-feature outputs."""
    if hasattr(output, "norm"):
        return output

    for attr in ("text_embeds", "text_embedding", "embeds"):
        value = getattr(output, attr, None)
        if value is not None and hasattr(value, "norm"):
            return value

    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None and hasattr(pooler_output, "norm"):
        projection = getattr(model, "text_projection", None)
        if projection is not None:
            try:
                return projection(pooler_output)
            except Exception:
                pass
        return pooler_output

    if isinstance(output, (tuple, list)):
        if len(output) > 1 and hasattr(output[1], "norm"):
            projection = getattr(model, "text_projection", None)
            if projection is not None:
                try:
                    return projection(output[1])
                except Exception:
                    pass
            return output[1]
        for value in output:
            if hasattr(value, "norm"):
                return value

    raise TypeError(f"Could not extract text feature tensor from {type(output).__name__}")


class CLIPObjectMatcher(_TextMatcherBase):
    """CLIP text encoder for object-class semantic matching."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", verbose: bool = True):
        super().__init__()
        if not HAS_CLIP:
            raise ImportError(
                "CLIP matching requires torch and transformers. "
                "Install: pip install torch transformers"
            )
        if verbose:
            print(f"  Loading CLIP text model (objects): {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def _prepare_text(self, label: str) -> str:
       return f"{label}"   

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            return_tensors="pt", max_length=77,
        ).to(self.device)
        with torch.no_grad():
            embs = self.model.get_text_features(**inputs)
            embs = _coerce_text_features_to_tensor(embs, self.model)
            embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        return embs.detach().cpu().numpy()

    def encode(self, text: str) -> np.ndarray:
        prompted_text = self._prepare_text(text)
        if prompted_text not in self._cache:
            self._cache[prompted_text] = self._encode_batch([prompted_text])[0]
        return self._cache[prompted_text]

    def precompute(self, texts: List[str]):
        new_texts = sorted(set(t for t in texts if t not in self._cache))
        if not new_texts:
            return
        batch_size = 256
        for i in range(0, len(new_texts), batch_size):
            chunk = new_texts[i:i + batch_size]
            embs = self._encode_batch(chunk)
            for t, e in zip(chunk, embs):
                self._cache[t] = e


class PredicateMatcher(_TextMatcherBase):
    """Sentence-BERT encoder for predicate semantic matching."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", verbose: bool = True):
        super().__init__()
        if not HAS_SBERT:
            raise ImportError(
                "sentence-transformers is required for soft matching. "
                "Install: pip install sentence-transformers"
            )
        if verbose:
            print(f"  Loading sentence-BERT model (predicates): {model_name}")
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        if text not in self._cache:
            emb = self.model.encode([text], normalize_embeddings=True)[0]
            self._cache[text] = emb
        return self._cache[text]

    def precompute(self, texts: List[str]):
        new_texts = sorted(set(t for t in texts if t not in self._cache))
        if new_texts:
            embs = self.model.encode(
                new_texts, normalize_embeddings=True,
                show_progress_bar=len(new_texts) > 100, batch_size=256,
            )
            for t, e in zip(new_texts, embs):
                self._cache[t] = e


class JinaPredicateMatcher(_TextMatcherBase):
    """Jina encoder for predicate semantic matching, matching Open3DSG eval."""

    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-en", verbose: bool = True):
        super().__init__()
        if not HAS_CLIP:
            raise ImportError(
                "Jina predicate matching requires torch and transformers. "
                "Install: pip install torch transformers"
            )
        if verbose:
            print(f"  Loading Jina model (predicates): {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            embs = self.model.encode(texts)
        embs = np.asarray(embs, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=-1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return embs / norms

    def encode(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self._encode_batch([text])[0]
        return self._cache[text]

    def precompute(self, texts: List[str]):
        new_texts = sorted(set(t for t in texts if t not in self._cache))
        if not new_texts:
            return
        batch_size = 256
        for i in range(0, len(new_texts), batch_size):
            chunk = new_texts[i:i + batch_size]
            embs = self._encode_batch(chunk)
            for t, e in zip(chunk, embs):
                self._cache[t] = e
