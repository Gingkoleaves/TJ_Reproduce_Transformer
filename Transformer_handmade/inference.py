import argparse
import warnings

import torch

from Transformer_handmade.config import TransformerConfig, get_config
from Transformer_handmade.data.my_tokenizer import BPETokenizer
from Transformer_handmade.model import Seq2SeqTransformer

warnings.filterwarnings("ignore", message=".*nested tensor.*prototype.*")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def load_model(
    checkpoint_path: str | None = None,
) -> tuple[Seq2SeqTransformer, BPETokenizer, BPETokenizer, TransformerConfig, torch.device]:
    base_config = get_config()
    ckpt_path = checkpoint_path or base_config.checkpoint_path
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config_payload = checkpoint.get("config", {})
    config = TransformerConfig(**config_payload) if config_payload else base_config
    device = resolve_device(config.device)

    tokenizer = BPETokenizer.load(config.tokenizer_path)
    src_tokenizer = tokenizer
    tgt_tokenizer = tokenizer

    model = Seq2SeqTransformer(config, checkpoint["src_vocab_size"], checkpoint["tgt_vocab_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_pad_ids(src_tokenizer.pad_id, tgt_tokenizer.pad_id)
    model.eval()
    return model, src_tokenizer, tgt_tokenizer, config, device


@torch.no_grad()
def translate(text: str, checkpoint_path: str | None = None, use_beam: bool = False) -> str:
    model, src_tokenizer, tgt_tokenizer, config, device = load_model(checkpoint_path)
    src_ids = src_tokenizer.encode(text)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_padding_mask = src.eq(src_tokenizer.pad_id)

    if use_beam:
        generated = model.beam_search_decode(
            src=src,
            src_padding_mask=src_padding_mask,
            bos_id=tgt_tokenizer.bos_id,
            eos_id=tgt_tokenizer.eos_id,
            max_len=config.max_seq_len,
            beam_size=config.beam_size,
            length_penalty=config.length_penalty,
        )
    else:
        generated = model.greedy_decode(
            src=src,
            src_padding_mask=src_padding_mask,
            bos_id=tgt_tokenizer.bos_id,
            eos_id=tgt_tokenizer.eos_id,
            max_len=config.max_seq_len,
        )
    return tgt_tokenizer.decode(generated[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run decoding with the PyTorch transformer.")
    parser.add_argument("--text", required=True, help="Source sentence to translate.")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt file.")
    parser.add_argument("--beam", action="store_true", help="Use beam search (default: greedy).")
    args = parser.parse_args()
    print(translate(args.text, args.checkpoint, args.beam))
