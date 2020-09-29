import copy
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torchcrf import CRF


class CustomNLLLoss(torch.nn.Module):
    def __init__(self):
        super(CustomNLLLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, targets):
        logs = self.log_softmax(x)
        zeros = []
        ones = []
        for i in range(len(targets)):
            current = targets[i]

            if current == 1:
                ones.append(logs[i][targets[i]])

            elif current == 0:
                zeros.append(logs[i][targets[i]])

        if len(ones) > 0:
            r = -sum(ones) / len(ones)
        else:
            r = 0
        if len(zeros) > 0:
            p = -sum(zeros) / len(zeros)
        else:
            p = 0

        return p + r


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, config):
        super(BiLSTM_CRF, self).__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=1, bidirectional=True)
        self.NUM_TAGS = 2

        self.crf = CRF(self.NUM_TAGS)
        self.hidden2tag = nn.Linear(hidden_dim, self.NUM_TAGS)

    def init_hidden(self, batch_size):
        if not self.config.cuda:
            return (torch.randn(2, batch_size, self.hidden_dim),
                    torch.randn(2, batch_size, self.hidden_dim))
        else:
            return (torch.randn(2, batch_size, self.hidden_dim).cuda(),
                    torch.randn(2, batch_size, self.hidden_dim).cuda())

    def _get_lstm_features(self, embeds):
        self.hidden = self.init_hidden(embeds.size(0))
        embeds = embeds.permute(1, 0, 2)
        lstm_out, self.lstm_hidden = self.lstm(embeds, self.hidden)
        lstm_out = (lstm_out[:, :, :self.embedding_dim] + lstm_out[:, :, self.embedding_dim:])
        lstm_out = lstm_out.permute(1, 0, 2)
        embeds = embeds.permute(1, 0, 2)
        lstm_feats = self.hidden2tag(embeds + lstm_out)
        return lstm_feats

    def neg_log_likelihood(self, embeds, tags, mask=None):
        feats = self._get_lstm_features(embeds)
        feats = feats.permute(1, 0, 2)
        return -self.crf(feats, tags, mask=mask)

    def forward(self, embeds):
        lstm_feats = self._get_lstm_features(embeds)
        lstm_feats = lstm_feats.permute(1, 0, 2)
        return lstm_feats, self.crf.decode(lstm_feats)


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()

        self.emb_layers.append(
            nn.Embedding(n_token, d_embed)
        )
        if d_proj != d_embed:
            self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))

    def forward(self, inp):
        embed = self.emb_layers[0](inp)
        embed.mul_(self.emb_scale)

        return embed


class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 keep_order=False):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.relu = torch.nn.ReLU()

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(
                        nn.Parameter(torch.Tensor(d_proj, d_embed))
                    )
                else:
                    self.out_projs.append(None)

            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val ** i)

                self.out_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i))
                )

                self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))

        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)

        return logit

    def forward(self, hidden, target, keep_order=False):
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        '''

        if hidden.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                                        self.out_layers[0].bias, self.out_projs[0])
            nll = -F.log_softmax(logit, dim=-1) \
                .gather(1, target.unsqueeze(1)).squeeze(1)

        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target,
                                   dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]

                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    logprob_i = head_logprob_i[:, -i] \
                                + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)

                if (hasattr(self, 'keep_order') and self.keep_order) or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset:offset + logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)

        return nll


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        return pos_emb[None, :, :].expand(bsz, -1, -1).contiguous()


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.scale = scale
        self.c_proj = Conv1D(n_state, nx)
        self.d_proj = Conv1D(n_state, 4)
        self.n_ctx = config.n_ctx
        self.v = nn.Linear(config.n_ctx, config.n_embd // config.n_head, bias=False)
        self.dropout = nn.Dropout(0.3)
        self.masked_lm = config.masked_lm

    def _attn(self, q, k, v, k_r):
        w = torch.matmul(q, k)
        w_r = torch.matmul(q, k_r)
        w = w + w_r

        w = w / math.sqrt(v.size(-1))

        # add mask if not masked lm
        if not self.masked_lm:
            nd, ns = w.size(-2), w.size(-1)
            b = self.bias[:, :, ns - nd:ns, :ns]
            w = w * b + -1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)

        return torch.matmul(w, v), w

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, key_r):
        query = self.split_heads(query)
        query = query[:, :, -self.n_ctx:, :]
        key = self.split_heads(key, k=True)
        key_r = self.split_heads(key_r, k=True)
        value = self.split_heads(value)
        a, att = self._attn(query, key, value, key_r)
        a = self.merge_heads(a)

        return a, att


class MLP(nn.Module):
    def __init__(self, n_state, config):
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class EncodingBlock(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(EncodingBlock, self).__init__()
        nx = config.n_embd
        self.split_size = nx
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.c_attn = Conv1D(nx * 3, nx)
        self.r_attn = Conv1D(nx * 3, nx)
        self.tag_attn = Conv1D(nx * 3, nx)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(nx * 4, config)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, pos_emb):
        enc_x = self.ln_1(x)
        enc_x = self.c_attn(enc_x)
        query, key, value = enc_x.split(self.split_size, dim=2)

        enc_pos = self.r_attn(pos_emb)
        _, key_r, _ = enc_pos.split(self.split_size, dim=2)

        a, att = self.attn(query, key, value, key_r)
        x = x + a
        m = self.ln_2(x)
        m = self.mlp(m)
        m = self.dropout(m)
        x = x + m
        return x, att


class TransformerHead(nn.Module):

    def __init__(self, model_embeddings, config):
        super(TransformerHead, self).__init__()
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.div_val = 1
        self.sigmoid = nn.Sigmoid()

        self.adaptive_softmax = config.adaptive
        self.classification = config.classification
        self.masked_lm = config.masked_lm
        self.rnn = config.rnn
        self.crf = config.crf
        self.config = config

        emb_weights = model_embeddings.emb_layers[0].weight
        embed_shape = model_embeddings.emb_layers[0].weight.shape

        if self.classification:
            self.classification = True
            self.loss_function = CustomNLLLoss()
            self.decoder = nn.Linear(embed_shape[1], 2, bias=True)
            self.num_layers = 2

            if self.rnn:
                self.lstm = nn.LSTM(embed_shape[1], embed_shape[1], dropout=0.3, num_layers=self.num_layers,
                                    bidirectional=True)
                self.lstm_dropout = nn.Dropout(0.3)
            elif self.crf:
                self.crf = BiLSTM_CRF(config.vocab_size, config.n_embd, config.n_embd, config)

        else:
            self.loss_function = nn.CrossEntropyLoss()
            self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=True)
            self.decoder.weight = emb_weights  # Tied weights

        if self.adaptive_softmax:

            n_token = config.vocab_size
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, self.n_embd, self.n_embd,
                                                    config.cutoffs, div_val=self.div_val)

            for i in range(len(self.crit.out_layers)):
                self.crit.out_layers[i].weight = model_embeddings.emb_layers[i].weight

            if config.tie_projs:
                for i, tie_proj in enumerate(config.tie_projs):

                    if tie_proj and self.div_val != 1:
                        self.crit.out_projs[i] = model_embeddings.emb_projs[i]

    def init_hidden(self, batch_size):
        if self.config.cuda:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.n_embd).cuda())
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.n_embd).cuda())
        else:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.n_embd))
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.n_embd))
        return (h0, c0)

    def forward(self, hidden_state, target, masked_idx, test=False, predict=False):

        if self.classification:
            if self.crf:
                mask = target.clone()
                mask[mask > 0] = 1
                mask = mask.type(torch.uint8)
                mask = mask.permute(1, 0)
                target[target == 1] = 0
                target[target > 1] = 1
                target = target.permute(1, 0)

                if not test:
                    loss = self.crf.neg_log_likelihood(hidden_state, target, mask=mask)
                    return loss
                elif predict:
                    logits, predictions = self.crf(hidden_state)
                    logits = logits.permute(1, 0, 2)
                    return logits, predictions
                elif test:
                    loss = self.crf.neg_log_likelihood(hidden_state, target, mask=mask)
                    logits, predictions = self.crf(hidden_state)
                    logits = logits.permute(1, 0, 2)
                    return loss, logits, predictions
            else:
                if self.rnn:
                    lstm_hidden = self.init_hidden(hidden_state.size(0))
                    embeddings = self.lstm_dropout(hidden_state).permute(1, 0, 2)
                    lstm_out, self.lstm_hidden = self.lstm(embeddings, lstm_hidden)
                    lstm_out = (lstm_out[:, :, :self.n_embd] + lstm_out[:, :, self.n_embd:])
                    lstm_out = lstm_out.permute(1, 0, 2)
                    hidden_state = hidden_state + lstm_out

                logits = self.decoder(self.dropout(self.relu(hidden_state)))
                if predict:
                    return logits

                active_loss = target.contiguous().view(-1) > 0
                active_logits = logits.contiguous().view(-1, logits.size(-1)).squeeze(1)[active_loss]

                active_targets = target.contiguous().view(-1) - 1

                active_targets = active_targets[active_loss]

                binary_targets = active_targets.clone()
                binary_targets[binary_targets > 0] = 1

                binary_loss = self.loss_function(active_logits, binary_targets)

                loss = binary_loss

                if test:
                    return loss, logits
                return loss

        else:
            # use adaptive softmax (including standard softmax)
            if self.adaptive_softmax:
                if test:
                    logits = self.dropout(self.decoder(hidden_state))

                if self.masked_lm:
                    hidden_state = hidden_state[masked_idx]
                    target = target[masked_idx]

                loss = self.crit(hidden_state.contiguous().view(-1, hidden_state.contiguous().size(-1)),
                                 target.contiguous().view(-1), test)
            else:
                logits = self.dropout(self.decoder(hidden_state))
                loss = self.loss_function(logits.contiguous().view(-1, logits.size(-1)), target.contiguous().view(-1))

            if test:
                return loss, logits
            return loss


class TransformerModel(nn.Module):

    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config

        self.wte = AdaptiveEmbedding(config.vocab_size, config.n_embd, config.n_ctx, config.cutoffs)

        if config.POS_tags:
            self.pos_wte = AdaptiveEmbedding(config.vocab_size, config.n_embd, config.n_ctx, config.cutoffs)

        self.n_layer = config.n_layer

        encodingblock = EncodingBlock(config.n_ctx, config, scale=True)
        self.enc_h = nn.ModuleList([copy.deepcopy(encodingblock) for _ in range(config.n_layer)])

        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.head = TransformerHead(self.wte, config)
        self.apply(self.init_weights)
        self.pe = PositionalEmbedding(config.n_embd)

        self.output_pe = PositionalEmbedding(config.n_embd)
        self.output_wte = AdaptiveEmbedding(config.vocab_size, config.n_embd, config.n_ctx, config.cutoffs)

    def set_tied(self):
        self.head.set_embeddings_weights(self.transformer.wte.weight)

    def init_weights(self, module):

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, input_pos=None, lm_labels=None, masked_idx=None, test=False,
                predict=False):

        input_ids = input_ids.view(-1, input_ids.size(-1))
        inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds

        if input_pos is not None:
            input_pos = input_pos.view(-1, input_ids.size(-1))
            pos_tag_embeds = self.pos_wte(input_pos)
            hidden_states = hidden_states + pos_tag_embeds

        param = next(self.parameters())
        klen = input_ids.size(1)
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=param.device)
        pos_emb = self.pe(pos_seq, bsz=input_ids.size(0))

        for block in self.enc_h:
            hidden_states, att = block(hidden_states, pos_emb)

        hidden_states = self.ln_f(hidden_states)

        if predict and self.config.classification:
            logits = self.head(hidden_states, lm_labels, masked_idx, predict=predict)
            return logits

        if test:
            if self.config.classification:
                if self.config.crf:
                    loss, logits, embedding_logits = self.head(hidden_states, lm_labels, masked_idx, test=test)
                    return loss, logits, embedding_logits, att
                else:
                    loss, logits = self.head(hidden_states, lm_labels, masked_idx, test=test)
                    return loss, logits, att
            else:
                loss, logits = self.head(hidden_states, lm_labels, masked_idx, test=test)
                return loss, logits
        loss = self.head(hidden_states, lm_labels, masked_idx, test=test)

        return loss





