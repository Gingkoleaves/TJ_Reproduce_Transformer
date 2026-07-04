import math

import torch
from torch import nn

from Transformer_handmade.config import TransformerConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


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

        self.src_embedding = nn.Embedding(src_vocab_size, config.d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.dropout, config.max_seq_len)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.h,
            num_encoder_layers=config.N,
            num_decoder_layers=config.N,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(config.d_model, tgt_vocab_size)

        if share_embeddings:
            # Tie embedding weights (paper §3.4, Table 3 row E):
            #   share src_emb, tgt_emb, and output projection.
            assert src_vocab_size == tgt_vocab_size, (
                f"Shared embeddings require src_vocab_size == tgt_vocab_size, "
                f"got {src_vocab_size} != {tgt_vocab_size}"
            )
            self.tgt_embedding.weight = self.src_embedding.weight
            self.generator.weight = self.src_embedding.weight

        self._init_params()

    def _init_params(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def set_pad_ids(self, src_pad_id: int, tgt_pad_id: int) -> None:
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

    def embed_src(self, src: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding(self.src_embedding(src) * math.sqrt(self.config.d_model))

    def embed_tgt(self, tgt: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.config.d_model))

    def generate_square_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        src_padding_mask: torch.Tensor | None = None,
        tgt_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1), src.device)
        output = self.transformer(
            src=self.embed_src(src),
            tgt=self.embed_tgt(tgt_input),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.generator(output)

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
        memory = self.transformer.encoder(
            self.embed_src(src),
            src_key_padding_mask=src_padding_mask,
        )
        generated = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)

        for _ in range(max_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(generated.size(1), src.device)
            decoder_output = self.transformer.decoder(
                self.embed_tgt(generated),
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=generated.eq(self.tgt_pad_id),
                memory_key_padding_mask=src_padding_mask,
            )
            next_token_logits = self.generator(decoder_output[:, -1])
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

        memory = self.transformer.encoder(
            self.embed_src(src),
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
            tgt_mask = self.generate_square_subsequent_mask(T, device)
            dec_out = self.transformer.decoder(
                self.embed_tgt(seqs),
                memory.expand(B, -1, -1),
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=seqs.eq(self.tgt_pad_id),
                memory_key_padding_mask=src_padding_mask.expand(B, -1),
            )  # (B, T, d_model)

            log_probs = torch.log_softmax(self.generator(dec_out[:, -1]), dim=-1)  # (B, V)
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
