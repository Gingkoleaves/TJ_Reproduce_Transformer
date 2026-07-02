import math

import torch
from torch import nn

from Transformer_handmade import config
from Transformer_handmade.config import TransformerConfig
from Transformer_handmade.model.encoder import Seq2SeqEncoder
from Transformer_handmade.model.decoder import Seq2SeqDecoder
from Transformer_handmade.model.embedding import Seq2SeqEmbedding, PositionalEncoding

class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        src_vocab_size: int,
        tgt_vocab_size: int,
        share_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.src_pad_id = 0
        self.tgt_pad_id = 0
        self.share_embeddings = share_embeddings

        self.embeddings = Seq2SeqEmbedding(
            config, src_vocab_size, tgt_vocab_size, share_embeddings=share_embeddings
        )
        self.encoder = Seq2SeqEncoder(config)
        self.decoder = Seq2SeqDecoder(config)

        self._init_params()

    def _init_params(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def set_pad_ids(self, src_pad_id: int, tgt_pad_id: int) -> None:
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

    # ------------------------------------------------------------------
    # Convenience accessors (used by tests / inference)
    # ------------------------------------------------------------------

    @property
    def src_embedding(self) -> nn.Embedding:
        return self.embeddings.src_embedding

    @property
    def tgt_embedding(self) -> nn.Embedding:
        return self.embeddings.tgt_embedding

    @property
    def generator(self) -> nn.Linear:
        return self.embeddings.generator

    @staticmethod
    def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
        return Seq2SeqDecoder.generate_square_subsequent_mask(size, device)

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        src_padding_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encoder(self.embeddings.embed_src(src), src_key_padding_mask=src_padding_mask)
        output = self.decoder(
            self.embeddings.embed_tgt(tgt_input),
            memory,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.embeddings.generator(output)

    # ------------------------------------------------------------------
    # Greedy decoding (fast, for validation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_padding_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> torch.Tensor:
        memory = self.encoder(
            self.embeddings.embed_src(src),
            src_key_padding_mask=src_padding_mask,
        )
        generated = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)

        for _ in range(max_len - 1):
            decoder_output = self.decoder(
                self.embeddings.embed_tgt(generated),
                memory,
                tgt_key_padding_mask=generated.eq(self.tgt_pad_id),
                memory_key_padding_mask=src_padding_mask,
            )
            next_token_logits = self.embeddings.generator(decoder_output[:, -1])
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.squeeze(1) == eos_id):
                break

        return generated

    # ------------------------------------------------------------------
    # Beam search decoding (paper: beam=4, length-penalty α=0.6)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def beam_search_decode(
        self,
        src: torch.Tensor,
        src_padding_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        beam_size: int = 4,
        length_penalty: float = 0.6,
    ) -> torch.Tensor:
        """Batched beam search (src: 1 × S).

        One decoder call per step with all active beams as a batch,
        giving beam_size× speedup over the sequential version.
        """
        assert src.size(0) == 1, "beam_search_decode expects batch_size=1"
        device = src.device

        memory = self.encoder(
            self.embeddings.embed_src(src),
            src_key_padding_mask=src_padding_mask,
        )  # (1, S, d_model)

        # seqs[i]: 1-D token sequence for beam i (all beams same length at each step)
        # scores[i]: accumulated log-prob for beam i
        seqs = torch.full((1, 1), bos_id, dtype=torch.long, device=device)   # (1, 1)
        scores = torch.zeros(1, dtype=torch.float, device=device)             # (1,)
        completed: list[tuple[torch.Tensor, float]] = []

        for _ in range(max_len - 1):
            B, T = seqs.shape

            # Single batched decoder call for all B active beams
            dec_out = self.decoder(
                self.embeddings.embed_tgt(seqs),
                memory.expand(B, -1, -1),
                tgt_key_padding_mask=seqs.eq(self.tgt_pad_id),
                memory_key_padding_mask=src_padding_mask.expand(B, -1),
            )  # (B, T, d_model)

            log_probs = torch.log_softmax(self.embeddings.generator(dec_out[:, -1]), dim=-1)  # (B, V)
            topk_lp, topk_ids = torch.topk(log_probs, beam_size, dim=-1)          # (B, beam_size)

            # All B×beam_size candidate scores
            cand_scores = scores.unsqueeze(1) + topk_lp                            # (B, beam_size)
            lp = ((5.0 + T + 1) / 6.0) ** length_penalty
            flat_normed = (cand_scores / lp).view(-1)
            flat_scores = cand_scores.view(-1)
            flat_ids = topk_ids.view(-1)
            parents = torch.arange(B, device=device).unsqueeze(1).expand(-1, beam_size).reshape(-1)

            top_k_idx = torch.topk(flat_normed, min(beam_size, flat_normed.numel())).indices

            next_seqs: list[torch.Tensor] = []
            next_scores: list[float] = []
            for idx in top_k_idx.tolist():
                p = parents[idx].item()
                tok = flat_ids[idx].item()
                sc = flat_scores[idx].item()
                new_seq = torch.cat([seqs[p], seqs.new_tensor([tok])])  # (T+1,)
                if tok == eos_id:
                    completed.append((new_seq, sc))
                else:
                    next_seqs.append(new_seq)
                    next_scores.append(sc)

            if not next_seqs:
                break

            seqs = torch.stack(next_seqs)
            scores = seqs.new_tensor(next_scores, dtype=torch.float)
        else:
            # Loop exhausted max_len without all beams finishing
            for j in range(seqs.size(0)):
                completed.append((seqs[j], scores[j].item()))

        if not completed:
            return seqs[0:1] if seqs.size(0) > 0 else seqs.new_full((1, 1), eos_id)

        def _normed_score(c: tuple[torch.Tensor, float]) -> float:
            seq, sc = c
            return sc / ((5.0 + seq.size(0)) / 6.0) ** length_penalty

        best_seq, _ = max(completed, key=_normed_score)
        return best_seq.unsqueeze(0)
