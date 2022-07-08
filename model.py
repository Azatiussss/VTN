import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class Config:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VTN(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.representation_size = 512
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        self.encoder_blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.mu_fc = nn.Linear(config.n_embd, self.representation_size)
        self.sigma_fc = nn.Linear(config.n_embd, self.representation_size)

        self.lstm = nn.LSTM(input_size=config.n_embd, hidden_size=self.representation_size,
                          num_layers=2, batch_first=True)

        self.mu_prior_fc = nn.Linear(config.n_embd, self.representation_size)
        self.sigma_prior_fc = nn.Linear(config.n_embd, self.representation_size)

        self.decoder_blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.beta = torch.FloatTensor([0])

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def reparametrize(self, mu, log_sigma, mu_prior, log_sigma_prior):
        batch_size = mu.size(0)
        device = mu.device

        standard_normal = torch.randn((mu.shape), device=device)
        z = mu + standard_normal * torch.exp(log_sigma)

        kl_divergence = torch.mean(torch.sum(log_sigma_prior - log_sigma + \
                        (torch.exp(log_sigma) ** 2 + (mu - mu_prior)**2) / (2 * torch.exp(log_sigma_prior)** 2) - 0.5, dim=-1), dim=-1)

        return z, kl_divergence

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.LSTM)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif (pn.endswith('weight') or pn.endswith('l0') or pn.endswith('l1')) and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward_encoder(self, x):
        token_embeddings = self.tok_emb(x)
        x = self.drop(token_embeddings)
        x = self.encoder_blocks(x)
        return token_embeddings, x

    def forward_decoder(self, z):
        z = self.decoder_blocks(z)
        z = self.ln_f(z)
        logits = self.head(z)
        return logits

    def forward(self, idx, targets=None, pad_token=-100):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        embs, x = self.forward_encoder(idx)
        mu = self.mu_fc(x)
        sigma = self.sigma_fc(x)

        lstm_output, _ = self.lstm(embs)
        mu_prior = self.mu_prior_fc(lstm_output)
        sigma_prior = self.sigma_prior_fc(lstm_output)

        z, kl_divergence = self.reparametrize(mu, sigma, torch.zeros_like(mu), torch.zeros_like(mu))
        logits = self.forward_decoder(z)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token) + self.beta.to(kl_divergence.device) * kl_divergence.mean(dim=0)

        return logits, loss