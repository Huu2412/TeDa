"""BPE tokenizer for CTC targets (SentencePiece, 1024 units).

Id layout: 0 = <blank> (CTC blank), 1 = <unk>, 2.. = BPE pieces. No <sos>/<eos> (CTC only).
Trained on the *normalised* training transcripts only (never on valid/test).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

BLANK_ID = 0
UNK_ID = 1


def train_bpe(texts: Iterable[str], model_prefix: str | Path, vocab_size: int = 1024, workdir: str | Path | None = None) -> Path:
    import sentencepiece as spm

    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    corpus = Path(workdir or model_prefix.parent) / (model_prefix.name + ".corpus.txt")
    n = 0
    with open(corpus, "w", encoding="utf-8") as f:
        for t in texts:
            t = " ".join(str(t).split())
            if t:
                f.write(t + "\n")
                n += 1
    if n == 0:
        raise ValueError("no training text given to train_bpe")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,          # keep every Vietnamese diacritic letter
        pad_id=BLANK_ID, pad_piece="<blank>",
        unk_id=UNK_ID, unk_piece="<unk>",
        bos_id=-1, eos_id=-1,
        normalization_rule_name="identity",  # text is already normalised by teda.data.text_norm
        remove_extra_whitespaces=False,
        hard_vocab_limit=False,
    )
    corpus.unlink(missing_ok=True)
    return model_prefix.with_suffix(".model")


class BPETokenizer:
    def __init__(self, model_path: str | Path):
        import sentencepiece as spm

        if not Path(model_path).exists():
            raise FileNotFoundError(f"tokenizer model not found: {model_path} (run scripts/build_tokenizer.py)")
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        if self.sp.pad_id() != BLANK_ID or self.sp.unk_id() != UNK_ID:
            raise ValueError("tokenizer must have <blank>=0 and <unk>=1; rebuild with scripts/build_tokenizer.py")

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: Iterable[int]) -> str:
        ids = [int(i) for i in ids if int(i) not in (BLANK_ID,)]
        return " ".join(self.sp.decode(ids).split())

    def unk_rate(self, texts: Iterable[str]) -> float:
        n = u = 0
        for t in texts:
            ids = self.encode(t)
            n += len(ids)
            u += sum(1 for i in ids if i == UNK_ID)
        return u / max(n, 1)
