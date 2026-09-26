import sys

import sentencepiece as spm

# Windows consoles default to a legacy codepage (e.g. cp1252) that can't
# print SentencePiece's "▁" whitespace marker or other unicode pieces.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TOKENIZER_MODEL = "tokenizer/tokenizer.model"

TEST_SENTENCES = [
    "The cat is sleeping on the mat",
    "public class LoanService",
    "Machine learning is a subset of AI",
]


def main():
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_MODEL)

    all_passed = True
    for sentence in TEST_SENTENCES:
        ids = sp.encode(sentence, out_type=int)
        pieces = sp.encode(sentence, out_type=str)
        decoded = sp.decode(ids)

        print(f"\nSentence: {sentence!r}")
        print(f"  Tokens:  {pieces}")
        print(f"  IDs:     {ids}")
        print(f"  Decoded: {decoded!r}")

        roundtrip_ok = decoded.strip() == sentence.strip()
        print(f"  Roundtrip match: {roundtrip_ok}")
        if not roundtrip_ok:
            all_passed = False

    print()
    if all_passed:
        print("ALL ROUNDTRIP TESTS PASSED")
    else:
        print("ROUNDTRIP TEST FAILURE DETECTED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
