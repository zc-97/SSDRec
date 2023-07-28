from HsdUtils.utils import get_model
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from torch.nn.init import xavier_uniform_, xavier_normal_
import math


class HSD(SequentialRecommender):

    def __init__(self, config, dataset):
        super(HSD, self).__init__(config, dataset)

        # load parameters info
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.our_ae_drop_out = config['our_ae_drop_out']
        self.our_att_drop_out = config['our_att_drop_out']
        self.n_users = dataset.num(self.USER_ID)  # Compatible with the latest version of RecBole
        # self.to = config['device']
        self.tau = 100
        self.filter_drop_rate = 0.0

        self.user_embedding = nn.Embedding(self.n_users, self.hidden_size, padding_idx=0).to(self.device)
        # self.user_embedding = nn.Embedding(self.n_users, self.hidden_size, padding_idx=0)

        self.LayerNorm = nn.BatchNorm1d(self.hidden_size)
        self.emb_dropout = nn.Dropout(self.our_ae_drop_out)

        # 加入bi-LSTM
        self.rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            bidirectional=True,
            batch_first=True
        )

        self.conv = nn.Conv2d(self.max_seq_length, self.max_seq_length, (1, 2))
        self.seq_level_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 2, bias=False),
            nn.Sigmoid()
        )

        self.read_out = AttnReadout(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.hidden_size,
            session_len=self.max_seq_length,
            batch_norm=True,
            feat_drop=self.our_att_drop_out,
            activation=nn.PReLU(self.hidden_size),
        )

        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=1)
        self.binary_softmax = nn.Softmax(dim=-1)

        self.loss_fuc = nn.CrossEntropyLoss()

        # 初始化global graphs
        # self.g_user_item, self.g_user_item_t, self.g_pos_item_in, self.g_pos_item_out, self.g_neg_item, self.g_pos_user, self.g_neg_user = graphs
        #
        # self.graph_relu = nn.ReLU()

        self.apply(self._init_weights)

        # 初始化sub_model
        self.sub_model = get_model(config['sub_model'])(config, dataset).to(config['device'])
        self.sub_model_name = config['sub_model']
        self.item_embedding = self.sub_model.item_embedding

        if config['load_pre_train_emb'] is not None and config['load_pre_train_emb']:
            checkpoint_file = config['pre_train_model_dict'][config['dataset']][config['sub_model']]
            checkpoint = torch.load(checkpoint_file)
            if config['sub_model'] == 'DSAN':
                embedding_weight = checkpoint['state_dict']['embedding.weight']
                self.item_embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
            else:
                item_embedding_weight = checkpoint['state_dict']['item_embedding.weight']
                self.item_embedding = nn.Embedding.from_pretrained(item_embedding_weight, freeze=False).to('cuda:0')

        if config['sub_model'] == 'BERT4Rec':
            self.dna = DNA(config, dataset, self.item_embedding.weight[:self.n_items],
                                 self.user_embedding.weight[:]).to(self.device)
            # self.dna = DNA(config,dataset,self.item_embedding.weight[:self.n_items],self.user_embedding.weight[:])
        else:
            self.dna = DNA(config, dataset, self.item_embedding.weight, self.user_embedding.weight[:]).to(
                self.device)
            # self.dna = DNA(config,dataset,self.item_embedding.weight,self.user_embedding.weight[:])

        # print('---------------')
        # print(self.item_embedding.weight.device)
        # print(self.device)
        # print('---------------')

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def method_name(self, generated_seq, generated_seq_emb):
        try:
            row_indexes, col_id = torch.where(generated_seq.gt(0))
            row_flag = row_indexes[0]
        except:
            row_flag = 0
        index_flag = -1
        col_index = []
        for row_index in row_indexes:
            if row_index == row_flag:
                index_flag += 1
                col_index.append(index_flag)
            else:
                index_flag = 0
                col_index.append(index_flag)
                row_flag = row_index
        col_index = torch.tensor(col_index, device=self.device)
        denoising_seq = torch.zeros_like(generated_seq)
        denoising_seq_emb = torch.zeros_like(generated_seq_emb)
        denoising_seq[row_indexes, col_index] = generated_seq[row_indexes, col_id]
        denoising_seq_emb[row_indexes, col_index, :] = generated_seq_emb[row_indexes, col_id, :]
        return denoising_seq, denoising_seq_emb

    def generate_pos_seq(self, item_seq, item_seq_len, item_level_score, seq_level_score, mask):
        item_emb = self.item_embedding(item_seq)
        mask = mask.squeeze()

        # Todo 可调整
        item_level_gumbel_softmax_rst = F.gumbel_softmax(item_level_score, tau=self.tau, hard=True)
        seq_level_gumbel_softmax_rst = F.gumbel_softmax(seq_level_score, tau=self.tau, hard=True)

        item_level_denoising_seq_flag = item_level_gumbel_softmax_rst[:, :, 1] * mask
        seq_level_denoising_seq_flag = seq_level_gumbel_softmax_rst[:, :, 1] * mask

        noisy_flag = item_level_denoising_seq_flag * seq_level_denoising_seq_flag

        pos_flag = (1 - noisy_flag) * mask

        pos_seq_emb = item_emb * pos_flag.unsqueeze(-1)

        pos_seq = item_seq * pos_flag
        pos_seq[pos_seq != pos_seq] = 0

        pos_seq_len = torch.sum(pos_flag, dim=-1)

        # 如果序列为0 stamp会报错，因此这里将序列长度为0 的保留第一个
        pos_seq_len[pos_seq_len.eq(0)] = 1
        # pos_seq[pos_seq_len.eq(0)] = torch.zeros_like()

        # TODO: [1,2,3,4] -> [1,0,3,4] or [1,3,4]
        clean_seq_percent = torch.sum(pos_seq_len, dim=0) / item_seq_len.sum() * 100
        denoising_seq, denoising_seq_emb = self.method_name(pos_seq, pos_seq_emb)
        pos_seq = denoising_seq
        pos_seq_emb = denoising_seq_emb
        neg_seq_len = torch.squeeze(item_seq_len)

        return pos_seq, pos_seq_emb, pos_seq_len.long(), item_seq, neg_seq_len, clean_seq_percent

    def seq_level_consistency(self, item_seq, item_seq_len, mask, train_flag=True):
        item_seq_emb_ori = self.item_embedding(item_seq)

        item_seq_emb = self.emb_dropout(item_seq_emb_ori) * mask

        encoder_item_seq_emb_bi_direction, (encoder_hidden, mm_) = self.rnn(item_seq_emb)

        '将最后一个时刻的hidden state两个方向加起来'
        encoder_hidden = (encoder_hidden[0] + encoder_hidden[1]).squeeze()

        # torch.Size([2048, 200, 64])
        rnn1_hidden = int(encoder_item_seq_emb_bi_direction.shape[-1] / 2)
        encoder_item_seq_emb = encoder_item_seq_emb_bi_direction[:, :, :rnn1_hidden] + \
                               encoder_item_seq_emb_bi_direction[:, :, rnn1_hidden:]

        # TODO: 尝试随即删除部分item来输入到双向LSTM中
        encoder_item_seq_emb = encoder_item_seq_emb * mask
        decoder_item_seq_emb_bi_direction, _ = self.rnn(encoder_item_seq_emb)
        rnn2_hidden = int(decoder_item_seq_emb_bi_direction.shape[-1] / 2)
        decoder_item_seq_emb = decoder_item_seq_emb_bi_direction[:, :, :rnn2_hidden] + \
                               decoder_item_seq_emb_bi_direction[:, :, rnn2_hidden:]

        element_wise_reconstruction_loss = 0

        if train_flag:
            loss_fct = nn.MSELoss(reduction='none')
            element_wise_reconstruction_loss = loss_fct(decoder_item_seq_emb * mask, item_seq_emb_ori * mask).sum(
                -1).sum(
                -1) / item_seq_len.squeeze()

        concat_shuffled_and_origin = torch.stack((decoder_item_seq_emb, item_seq_emb_ori), dim=-1)  # [B len 2xh]
        concat_shuffled_and_origin = self.conv(concat_shuffled_and_origin)  # [B len h 1]
        concat_shuffled_and_origin = torch.squeeze(concat_shuffled_and_origin)  # [B len h]
        concat_shuffled_and_origin = self.emb_dropout(concat_shuffled_and_origin)  # [B len h]
        concat_shuffled_and_origin = nn.ReLU(inplace=True)(concat_shuffled_and_origin)  # [B len h]

        reconstruct_score = self.seq_level_mlp(concat_shuffled_and_origin).squeeze()  # [B len 2]

        reconstruct_score = reconstruct_score * mask
        return element_wise_reconstruction_loss, reconstruct_score, encoder_item_seq_emb, encoder_hidden

    def item_level_consistency(self, item_seq_emb, target_embedding, mask):
        item_level_score, item_level_long_term_representation, item_level_seq_emb = self.read_out(item_seq_emb,
                                                                                                  target_embedding,
                                                                                                  mask)

        return item_level_score, item_level_long_term_representation, item_level_seq_emb

    def loss_filter(self, user, item_seq, item_seq_len, interaction, mask, train_flag):
        item_seq_emb = self.item_embedding(item_seq)  # [B, L, 1, H]
        item_seq_emb = (item_seq_emb * mask).unsqueeze(-2)
        user_emb = self.user_embedding(user).unsqueeze(-2).unsqueeze(-1)  # [B, 1, H, 1]

        pos_score = torch.matmul(item_seq_emb, user_emb)  # [B, L, 1]

        filter_drop_rate = self.filter_drop_rate

        if train_flag:
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items = neg_items.unsqueeze(-1).expand(item_seq.shape)
            neg_items_emb = self.item_embedding(neg_items)  # [B, L, 1, H]
            neg_items_emb = (neg_items_emb * mask).unsqueeze(-2)
            neg_score = torch.matmul(neg_items_emb, user_emb)  # [B, L, 1]
        else:
            neg_score = torch.zeros_like(pos_score)
            filter_drop_rate = 0.2

        loss = -torch.log(1e-10 + torch.sigmoid(pos_score - neg_score)).squeeze(-1).squeeze(-1)  # [B, L, 1]
        loss = loss * mask.squeeze(-1)
        loss_sorted, ind_sorted = torch.sort(loss, descending=False, dim=-1)
        num_remember = (filter_drop_rate * item_seq_len).squeeze(-1).long()

        loss_filter_flag = torch.zeros_like(item_seq)

        for index, filtered_item_num in enumerate(num_remember):
            loss_index = ind_sorted[index][-filtered_item_num:]
            loss_filter_flag[index][loss_index] = 1
            if filter_drop_rate != 0:
                loss[index][loss_index] *= 0
        loss_filter_flag = loss_filter_flag * mask.squeeze(-1)
        return loss, loss_filter_flag

    def forward(self, interaction, best_flag=False,train_flag=True):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN].unsqueeze(1)
        if train_flag:
            target_item = interaction[self.ITEM_ID].unsqueeze(1)
        else:
            '如果是验证和测试的时候，由于不能提前预知下一项，因此将训练集的最后一项视作target'
            target_item = item_seq.gather(1, item_seq_len - 1)
        user = interaction[self.USER_ID]

        # calculate the mask
        mask = torch.ones(item_seq.shape, dtype=torch.float, device=item_seq.device) * item_seq.gt(0)

        # size = [batchSize, user_num_per_batch, 1]
        mask = mask.unsqueeze(2)

        # print('***************')
        # print(item_seq.shape)
        # print(item_seq_len.shape)
        # print(user.shape)
        # print('***************')

        # 插入dna

        item_seq, item_seq_emb, item_seq_len, all_i, all_u = self.dna(item_seq, item_seq_len, user, train_flag,
                                                                      self.tau)
        if not train_flag:
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_emb = all_i[item_seq]
        # item_seq_emb = self.item_embedding(item_seq)
        # de_seq.long(), de_seq_emb_pos, de_seq_len.unsqueeze(-1), item_embs, user_embs

        element_wise_reconstruction_loss, seq_level_score, seq_level_encoder_item_emb, seq_level_seq_emb = self.seq_level_consistency(
            item_seq=item_seq,
            item_seq_len=item_seq_len,
            mask=mask,
            train_flag=train_flag
        )

        if train_flag:
            target_embedding = self.item_embedding(target_item)
        else:
            target_embedding = self.user_embedding(user.unsqueeze(-1))
        item_level_score, item_level_long_term_representation, item_level_seq_emb = self.item_level_consistency(
            item_seq_emb=item_seq_emb,
            target_embedding=target_embedding,
            mask=mask)

        '将item- and seq-level 的embedding作为 emb-level denoising 从而增强data-level denoising 效果'
        embedding_level_denoising_emb = item_level_seq_emb + seq_level_seq_emb

        loss, loss_filter_flag = self.loss_filter(user, item_seq, item_seq_len, interaction, mask, train_flag)

        pos_seq, pos_seq_emb, pos_seq_len, neg_seq, neg_seq_len, clean_seq_percent = self.generate_pos_seq(
            item_seq=item_seq,
            item_seq_len=item_seq_len,
            item_level_score=item_level_score,
            seq_level_score=seq_level_score,
            mask=mask,
        )

        return element_wise_reconstruction_loss, pos_seq, pos_seq_emb, pos_seq_len, neg_seq, neg_seq_len, embedding_level_denoising_emb, clean_seq_percent, loss.sum(
            -1).mean()

    def calculate_loss_curriculum(self, interaction, drop_rate, tau):
        self.tau = tau
        self.filter_drop_rate = 0.2 - drop_rate

        element_wise_reconstruction_loss, pos_seq, pos_seq_emb, pos_seq_len, neg_seq, neg_seq_len, embedding_level_denoising_emb, clean_seq_percent, loss_filter_loss = self.forward(
            interaction)
        'positive seq                  --- A'
        'embedding_level_denoising_emb --- B'
        'negative seq                  --- C'
        'min/max D(AB)-D(AC)'

        user = interaction[self.USER_ID] if self.sub_model_name == 'Caser' else None
        all_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token

        # using the denoisied embedding calculate predict loss
        sub_model_output = self.sub_model_forward(pos_seq, pos_seq_emb, pos_seq_len, user)
        seq_representation, delete_index = self.denoising_seq_gather(pos_seq, sub_model_output)
        scores = torch.matmul(seq_representation, all_items_emb.transpose(0, 1))  # [B, item_num]

        ind_update = self.cal_curriculum_batch_id(drop_rate, element_wise_reconstruction_loss)
        element_wise_reconstruction_loss_curriculum = element_wise_reconstruction_loss[ind_update]
        L_rec = element_wise_reconstruction_loss_curriculum.mean()

        target_item = interaction[self.ITEM_ID].unsqueeze(1)
        target_item = target_item.squeeze()
        generated_seq_loss = self.loss_fuc(scores[ind_update], target_item[ind_update])

        total_loss = L_rec + generated_seq_loss + loss_filter_loss
        return total_loss, clean_seq_percent, L_rec, generated_seq_loss

    def cal_curriculum_batch_id(self, drop_rate, element_wise_reconstruction_loss):
        loss_sorted, ind_sorted = torch.sort(element_wise_reconstruction_loss, descending=False)
        remember_rate = 1 - drop_rate
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]
        return ind_update

    def predict(self, interaction):
        user = interaction[self.USER_ID] if self.sub_model_name == 'Caser' else None
        element_wise_reconstruction_loss, pos_seq, generated_seq, denoising_seq_len, temp1_, temp2_, embedding_level_denoising_emb, clean_seq_percent, loss = self.forward(
            interaction,
            train_flag=False)
        seq_output = self.sub_model_forward(generated_seq, denoising_seq_len, user)

        seq_output, _ = self.denoising_seq_gather(generated_seq, seq_output)

        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def denoising_seq_gather(self, generated_seq, seq_output):
        generated_item_seq_len = torch.sum(generated_seq.gt(0), 1)

        # 算出长度为0的则是全为噪声项的序列，这里先记录其index
        delete_index = torch.where(generated_item_seq_len.eq(0))[0]
        # 将index减一 ，防止数组越上界
        generated_item_seq_len = generated_item_seq_len - 1
        # 将index为-1 的项置为0，防止数组越下界
        generated_item_seq_len = generated_item_seq_len * generated_item_seq_len.gt(0)
        if self.sub_model_name in ['SRGNN', 'GCSAN', 'Caser', 'NARM', 'DSAN', 'STAMP']:
            seq_output = seq_output
        elif self.sub_model_name == 'fmlp':
            seq_output = seq_output[:, -1, :]  # delete masked token
        else:
            seq_output = self.gather_indexes(seq_output, generated_item_seq_len)  # [B H]
        return seq_output, delete_index

    def sub_model_forward(self, generated_seq, pos_seq_emb, denoising_seq_len, user):
        if self.sub_model_name == 'BERT4Rec':
            seq_output = self.sub_model.forward_denoising(generated_seq, pos_seq_emb)
        elif self.sub_model_name == 'GRU4Rec':
            seq_output = self.sub_model.forward_denoising(generated_seq, pos_seq_emb)
        elif self.sub_model_name == 'SASRec':
            seq_output = self.sub_model.forward_denoising(generated_seq, pos_seq_emb)
        elif self.sub_model_name == 'Caser':
            seq_output = self.sub_model.forward_denoising(user, generated_seq, pos_seq_emb)
        elif self.sub_model_name == 'NARM':
            seq_output = self.sub_model.forward_denoising(generated_seq, denoising_seq_len, pos_seq_emb)
        elif self.sub_model_name == 'DSAN':
            seq_output, _ = self.sub_model.forward(generated_seq)
        elif self.sub_model_name == 'fmlp':
            seq_output = self.sub_model.forward(generated_seq)
        elif self.sub_model_name == 'STAMP':
            seq_output = self.sub_model.forward_denoising(generated_seq, denoising_seq_len, pos_seq_emb)
        else:
            raise ValueError(f'Sub_model [{self.sub_model_name}] not support.')
        return seq_output

    def full_sort_predict_ssd(self, interaction,best_flag):
        user = interaction[self.USER_ID] if self.sub_model_name == 'Caser' else None
        element_wise_reconstruction_loss, pos_seq, pos_seq_emb, denoising_seq_len, _, _, embedding_level_denoising_emb, pre, loss = self.forward(
            interaction,best_flag=best_flag,
            train_flag=False)
        seq_output = self.sub_model_forward(pos_seq, pos_seq_emb, denoising_seq_len, user)

        seq_output, _ = self.denoising_seq_gather(pos_seq, seq_output)

        test_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores, pre


class AttnReadout(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            session_len,
            batch_norm=True,
            feat_drop=0.0,
            activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(session_len) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 2, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim else None
        )
        self.mlp_n_ls = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feat, last_nodes, mask):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = feat * mask
        feat = self.feat_drop(feat)
        feat = feat * mask
        feat_u = self.fc_u(feat)
        feat_u = feat_u * mask
        feat_v = self.fc_v(last_nodes)  # (batch_size * embedding_size)

        e = self.fc_e(F.tanh(feat_u + feat_v)) * mask
        e = self.sigmoid(e)

        short_term = last_nodes.squeeze()

        e0, rst = self.get_long_term(e, feat, mask)
        fuse_long_short = torch.cat((rst, short_term), dim=-1)
        item_level_seq_representation = self.mlp_n_ls(self.feat_drop(fuse_long_short))

        score = e.squeeze()
        return score, rst, item_level_seq_representation

    def get_long_term(self, e, feat, mask):
        'e是2维的分数，rst是long-term representation'
        mask1 = (mask - 1) * 2e32
        e0 = e[:, :, 1] + mask1.squeeze()
        beta = self.softmax(e0)
        feat_norm = feat * beta.unsqueeze(-1)
        feat_norm = feat_norm * mask
        rst = torch.sum(feat_norm, dim=1)
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return e0, rst


class DNA(SequentialRecommender):
    # def __init__(self, config, dataset,item_emb,user_emb):
    def __init__(self, config, dataset, item_emb, user_emb):
        # super(DNA, self).__init__(config, dataset)
        # super().__init__()
        super(DNA, self).__init__(config, dataset)
        # load parameters info
        # print('++++++++++')
        # print(item_emb.device)
        # print(user_emb.device)
        # print('++++++++++++++')

        self.embedding_size = config["hidden_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        self.item_embedding = item_emb
        self.tau = 100
        self.n_users = dataset.user_num

        self.user_embedding = user_emb

        self.emb_dropout = nn.Dropout(self.dropout_prob)

        self.conv = nn.Conv2d(self.max_seq_length, self.max_seq_length, (1, 2))
        self.seq_level_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 2, bias=False),
            nn.Sigmoid()
        )

        self._init_graph(config)

        self.denoise = Denoising(self.embedding_size, self.hidden_size, True)
        self.Bi_LSTM = self.denoise.Bi_LSTM

        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.LSTM):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def _init_graph(self, config):
        self.gnn_layer = 1
        graphs = config['graphs']
        self.g_user_item, self.g_user_item_t, self.g_pos_item_in, self.g_pos_item_out, self.g_neg_item, self.g_pos_user, self.g_neg_user = graphs

        self.i_concatenation_w = nn.Parameter(torch.Tensor(self.gnn_layer * self.embedding_size, self.embedding_size))
        self.u_concatenation_w = nn.Parameter(torch.Tensor(self.gnn_layer * self.embedding_size, self.embedding_size))

        self.act = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.i_w = nn.Parameter(torch.Tensor(self.embedding_size, self.embedding_size))
        self.u_w = nn.Parameter(torch.Tensor(self.embedding_size, self.embedding_size))
        self.ii_w = nn.Parameter(torch.Tensor(2 * self.embedding_size, self.embedding_size))
        self.uu_w = nn.Parameter(torch.Tensor(2 * self.embedding_size, self.embedding_size))

        self.W1 = nn.Parameter(torch.Tensor(self.embedding_size, self.embedding_size))
        self.W2 = nn.Parameter(torch.Tensor(self.embedding_size, self.embedding_size))

        self.conv_layer = nn.Conv2d(1, 1, (1, 2), bias=True)
        self.conv_layer_user = nn.Conv2d(1, 1, (1, 2), bias=True)
        self.feat_drop = nn.Dropout(self.dropout_prob)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.LeakyReLU(0.2)

        xavier_uniform_(self.i_w)
        xavier_uniform_(self.u_w)
        xavier_uniform_(self.ii_w)
        xavier_uniform_(self.uu_w)
        xavier_uniform_(self.conv_layer.weight)
        xavier_uniform_(self.conv_layer_user.weight)
        xavier_uniform_(self.W1)
        xavier_uniform_(self.W2)

        # i_seq_graph: gru -> attention

    def _augmentation(self, item_seq, all_i_emb, item_seq_emb, item_seq_len, position, de_flag, tau=1e-3):
        # item_seq_emb_pos, item_seq_emb_neg = item_seq_embs
        op = SeqOperation(seq=item_seq, seq_emb=item_seq_emb, pos=position)
        prepare_seq = op.prepare_seq()
        lstm_out_pos = self.Bi_LSTM(item_seq_emb)
        query_fw, query_bw = op.get_queries(lstm_out_pos, self.hidden_size)
        # self.item_embedding.weight[:] item_num x dim
        item_fw, item_bw, item_1, item_2 = op.get_item(all_i_emb, query_fw, query_bw, tau)

        fw, bw = op.insert_item(item_fw, item_bw)
        aug_seq_emb = prepare_seq + fw + bw
        aug_seq_len = item_seq_len + 2
        aug_seq_emb, aug_seq_len, mask_len = self._augmentation_rule(op.pos_row, item_seq_emb, item_seq_len,
                                                                     aug_seq_emb, aug_seq_len, de_flag)
        aug_flag = (aug_seq_len - item_seq_len) / 2

        aug_seq = self._insert_items(item_seq, op.pos_row, mask_len, item_1, item_2)
        return aug_seq, aug_seq_emb, aug_seq_len, aug_flag

    def _insert_items(self, item_seq, position, mask_len, insert_fw, insert_bw):
        new_seq = torch.zeros(item_seq.shape).to(item_seq.device)
        for seq, pos in enumerate(position):
            if mask_len[seq] > 0:
                '移位'
                new_seq[seq, :pos] = item_seq[seq, :pos]
                new_seq[seq, pos + 1] = item_seq[seq, pos]
                new_seq[seq, pos + 3:] = item_seq[seq, pos + 1:-2]
                '插值'
                new_seq[seq, pos] = insert_fw[seq]
                new_seq[seq, pos + 2] = insert_bw[seq]
            else:
                new_seq[seq] = item_seq[seq]
        return new_seq.long()

    def _augmentation_rule(self, pos_idx, ori_seq, ori_len, aug_seq, aug_len, de_flag):
        # indicate whether the inserted position is correct

        # mask for embeddings
        mask_pos_emb = (pos_idx <= ori_len).type(ori_seq.dtype).unsqueeze(1).unsqueeze(1)
        mask_seq_emb = (ori_len <= self.max_seq_length - 2).type(ori_seq.dtype).unsqueeze(1).unsqueeze(1)
        mask_pos_emb = mask_pos_emb * mask_seq_emb * de_flag.unsqueeze(1).unsqueeze(1)

        # mask for lengths
        mask_pos_len = (pos_idx <= ori_len).type(ori_len.dtype)
        mask_seq_len = (ori_len <= self.max_seq_length - 2).type(ori_len.dtype)
        mask_pos_len = mask_pos_len * mask_seq_len * de_flag

        # final outputs
        aug_seq = aug_seq * mask_pos_emb + ori_seq * (1 - mask_pos_emb)
        aug_len = aug_len * mask_pos_len + ori_len * (1 - mask_pos_len)
        return aug_seq, aug_len, mask_pos_len

    def agg_layer(self):
        all_user_embeddings = []
        all_item_embeddings = []
        for i in range(self.gnn_layer):
            u_emb_list = [None] * 2
            i_emb_list = [None] * 2
            u_pos = self.userGNN(self.user_embedding, self.g_pos_user)
            i_pos = self.itemGNN(self.item_embedding, self.g_pos_item_in, self.g_pos_item_out)

            u_neg = self.userGNN(self.user_embedding, self.g_neg_user)
            i_neg = self.itemGNN(self.item_embedding, self.g_neg_item, self.g_neg_item)

            u_ui = torch.spmm(self.g_user_item, self.item_embedding)
            i_ui = torch.spmm(self.g_user_item_t, self.user_embedding)

            u_emb_list[0] = torch.cat((u_ui, u_pos), -1) @ self.uu_w
            i_emb_list[0] = torch.cat((i_ui, i_pos), -1) @ self.ii_w

            u_emb_list[1] = torch.cat((u_ui, u_neg), -1) @ self.uu_w
            i_emb_list[1] = torch.cat((i_ui, i_neg), -1) @ self.ii_w

            user_embeddings = torch.stack(u_emb_list, dim=0)
            item_embeddings = torch.stack(i_emb_list, dim=0)

            user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w))
            item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w))

            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)
        user_embedding = torch.cat(all_user_embeddings, dim=1)
        item_embedding = torch.cat(all_item_embeddings, dim=1)
        user_embedding = torch.matmul(user_embedding, self.u_concatenation_w)
        item_embedding = torch.matmul(item_embedding, self.i_concatenation_w)
        return user_embedding, item_embedding

    def userGNN(self, user_emb, u_in):
        # print('--------------')
        # # print(u_in.device)
        # print(user_emb.shape,user_emb.shape)
        # print('--------------')
        neighbor_feature = torch.spmm(u_in, user_emb)
        agg = torch.stack((neighbor_feature, user_emb), dim=2)  # n x dim x 3
        agg = torch.unsqueeze(agg, 1)
        out_conv = self.conv_layer_user(agg)
        emb = self.feat_drop(torch.squeeze(out_conv))
        return emb

    def itemGNN(self, item_emb, i_in, i_out):
        in_neighbor = torch.spmm(i_in, item_emb)
        out_neighbor = torch.spmm(i_out, item_emb)

        x_in = self.relu((item_emb * in_neighbor) @ self.W1)
        x_out = self.relu((item_emb * out_neighbor) @ self.W2)

        in_score = torch.squeeze(torch.sum((x_in / math.sqrt(self.embedding_size)), dim=1), 0)
        out_score = torch.squeeze(torch.sum((x_out / math.sqrt(self.embedding_size)), dim=1), 0)
        score = self.softmax(torch.stack((in_score, out_score), dim=1))
        score_in = torch.unsqueeze(score[:, 0], dim=-1)
        score_out = torch.unsqueeze(score[:, 1], dim=-1)
        neighbor = in_neighbor * score_in + out_neighbor * score_out
        agg = torch.stack((item_emb, neighbor), dim=2)
        agg = torch.unsqueeze(agg, 1)
        out_conv = self.conv_layer(agg)
        emb = self.feat_drop(torch.squeeze(out_conv))
        return emb

    def forward(self, item_seq, seq_len, user, is_train, tau):
        # item_seq = interaction[self.ITEM_SEQ]
        # seq_len = interaction[self.ITEM_SEQ_LEN]
        # user = interaction[self.USER_ID]
        seq_len = seq_len.squeeze()
        seq_mask = torch.ones(item_seq.shape, dtype=torch.float, device=item_seq.device) * item_seq.gt(0)
        masks = (seq_mask, seq_mask.unsqueeze(-1), seq_mask.unsqueeze(1))
        user_embs, item_embs = self.agg_layer()
        scaled_user_emb = (user_embs[user] / seq_len.unsqueeze(-1)).unsqueeze(1)
        # print('-------------------------')
        # print(user.shape)  # batch
        # print(item_seq.shape)  # batch x items
        # print(seq_mask.shape)  # batch x items
        # print(seq_len.shape)  # batch
        # print(item_embs[item_seq].shape)  # batch x items x dim
        # print(scaled_user_emb.shape)  # batch x 1 x dim
        # print('-------------------------')
        # assert 1==2
        seq_emb = item_embs[item_seq] + scaled_user_emb

        # print('-----------------')
        # print(seq_emb.shape)
        # print(seq_len.shape)
        # print('-----------------')
        position, _, _, de_flag = self.denoise.forward_1st(seq_emb, seq_len, masks, tau)

        aug_seq, aug_seq_emb, aug_seq_len, aug_flag = self._augmentation(item_seq, item_embs, seq_emb, seq_len,
                                                                         position, de_flag, tau)

        aug_seq_mask = torch.ones(aug_seq.shape, dtype=torch.float, device=item_seq.device) * aug_seq.gt(0)
        masks = (aug_seq_mask, aug_seq_mask.unsqueeze(-1), aug_seq_mask.unsqueeze(1))

        # print('-------------------------')
        # print(aug_seq.shape)  # batch
        # print(aug_seq_emb.shape)  # batch x items
        # print(aug_seq_len.shape)  # batch x items
        # print(aug_seq_mask.shape)  # batch x 1
        # print('-------------------------')

        _, de_seq_emb_pos, de_seq_len, de_flag_2nd, de_seq = self.denoise.forward_2nd(position, de_flag, aug_seq_emb,
                                                                                      aug_seq_len, masks,
                                                                                      tau, aug_seq)

        return de_seq.long(), de_seq_emb_pos, de_seq_len.unsqueeze(-1), item_embs, user_embs

    def get_insert_position(self, item_seq, item_seq_len):
        'No Implement!'
        position = torch.zeros(item_seq.shape[0], dtype=torch.int64).to(self.device)
        return position

    def get_insert_item(self, item_seq, item_seq_len):
        insert_fw = item_seq[:, 0]
        insert_bw = item_seq[:, 1]
        insert_fw = self.item_embedding(insert_fw)
        insert_bw = self.item_embedding(insert_bw)
        return insert_fw, insert_bw


class SeqOperation(nn.Module):
    def __init__(self, seq, seq_emb, pos):
        super(SeqOperation, self).__init__()
        self.seq = seq
        self.seq_emb = seq_emb  # batch x item x dim
        self.pos = pos
        self.pos_rc = torch.nonzero(self.pos)
        self.pos_row = torch.nonzero(self.pos)[:, 1]
        self.n = self.seq_emb.shape[1]  # item_num
        self.relu = nn.LeakyReLU(0.2)
        self.w = nn.Parameter(torch.Tensor(100, 100)).to(self.seq.device)

    def _value_select(self):
        mask_pos = (self.pos_row <= self.n - 3)
        left_list = []
        right_list = []
        zeros = torch.zeros_like(self.pos[0])
        for seq in range(self.seq_emb.shape[0]):
            mask = mask_pos[seq]
            if not mask:
                left = right = zeros
            else:
                p = self.pos_row[seq]
                pos = self.pos[seq]
                pos_value = pos[p]
                left_cols = pos[:p]
                right_cols = pos[p + 1:]
                left = torch.cat([left_cols + pos_value, pos_value.unsqueeze(0) * 0, right_cols], dim=0)
                right = torch.cat([pos[:p + 1] * 0, pos[p + 1:self.n - 2] + pos_value, pos[self.n - 2:] * 0], dim=0)
            left_list.append(left)
            right_list.append(right)
        left_pos = torch.stack(left_list, dim=0).unsqueeze(-1)
        right_pos = torch.stack(right_list, dim=0).unsqueeze(-1)
        return left_pos, right_pos

    def _value_roll(self, mid_seq, shift):
        'comments:'
        'shift=1:  seq x pos'
        'shift=2:  seq x right'
        return torch.roll(mid_seq, shifts=shift, dims=1)

    def prepare_seq(self):
        left_pos, right_pos = self._value_select()
        seq_left = self.seq_emb * left_pos
        seq_right = self.seq_emb * right_pos
        seq_pos = self.seq_emb * self.pos.unsqueeze(-1)
        seq_pos_r1 = self._value_roll(mid_seq=seq_pos, shift=1)
        seq_right_r2 = self._value_roll(mid_seq=seq_right, shift=2)
        new_seq = seq_left + seq_pos_r1 + seq_right_r2
        return new_seq

    def get_queries(self, lstm_outputs, hidden_size):
        hn, _ = lstm_outputs
        hn_fw = hn[:, :, :hidden_size]
        hn_bw = hn[:, :, hidden_size:]
        pos_fw = self._value_roll(self.pos, -1)
        pos_bw = self._value_roll(self.pos, 1)
        fw_query_zero = hn_fw * pos_fw.unsqueeze(-1)
        bw_query_zero = hn_bw * pos_bw.unsqueeze(-1)
        fw_query = torch.sum(fw_query_zero, 1)
        bw_query = torch.sum(bw_query_zero, 1)
        # return: batch x dim
        return fw_query, bw_query

    def get_item(self, item_embeddings, query_fw, query_bw, tau):
        # query_fw_pos, query_fw_neg = query_fw
        # query_bw_pos, query_bw_neg = query_bw
        #
        # all_pos_item, all_neg_item = item_embeddings

        pos_items = item_embeddings.t()
        sim_fw_pos = query_fw @ pos_items
        sim_bw_pos = query_bw @ pos_items
        sim_fw = sim_fw_pos  # batch x dim  &  dim x item_num  -> batch x item_num
        sim_bw = sim_bw_pos
        # query_fw_pos @ all_pos_item
        # query_bw @ all_neg_item

        item_pos_fw = F.gumbel_softmax(sim_fw, tau=tau, hard=True)  # batch x item_num [0,0,1,0,0]
        item_pos_bw = F.gumbel_softmax(sim_bw, tau=tau, hard=True)
        item_fw = item_pos_fw @ item_embeddings  # batch x item_num  &  item_num x dim  -> batch x dim
        item_bw = item_pos_bw @ item_embeddings
        # return: batch x item_num x dim
        item_1 = torch.argmax(item_pos_fw, dim=1)
        item_2 = torch.argmax(item_pos_bw, dim=1)  # batch / int64
        return item_fw.unsqueeze(1), item_bw.unsqueeze(1), item_1, item_2

    def insert_item(self, item_fw, item_bw):
        pos_fw = self.pos.unsqueeze(2)  # 前向插入
        pos_bw = self._value_roll(mid_seq=self.pos, shift=2).unsqueeze(2)  # 后向插入
        fw = item_fw * pos_fw
        bw = item_bw * pos_bw
        return fw, bw


class Denoising(nn.Module):
    def __init__(self, embedding_size, hidden_size, is_graph):
        super(Denoising, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.se_embedding = nn.Embedding(2, self.embedding_size)
        self.Bi_LSTM = nn.LSTM(
            input_size=embedding_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=False,
            batch_first=True,
            bidirectional=True
        )
        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.is_graph = is_graph

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        if isinstance(module, nn.LSTM):
            xavier_uniform_(module.weight_ih_l0)
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0_reverse)
            xavier_uniform_(module.weight_hh_l0_reverse)

    def forward_1st(self, seq_emb, seq_len, masks, tau, hard=True):
        mask, mask_row, mask_col = masks
        seq_and_mask_pos = (seq_emb, mask, mask_row, mask_col)

        seq_level_pos = self._seq_consistency(seq_and_mask_pos)
        item_level_pos = self._item_consistency(seq_and_mask_pos)
        # noise_score = seq_level_pos * torch.clamp(item_level_pos,max=1.0,min=-1.0)
        noise_score = 1 - (item_level_pos * seq_level_pos)
        noise_prob = F.gumbel_softmax(noise_score, tau=tau, hard=hard).unsqueeze(-1)  # batch x item_num x 1

        seq_emb_1st_tmp = seq_emb  # batch x item_num x dim
        seq_len_1st_temp = seq_len - 1  # batch
        # input: batch x item_num x 1, batch x item_num x dim, batch, batch x item_num x dim, batch
        noise_pos = torch.nonzero(noise_prob.squeeze())[:, 1]  # batch
        if noise_pos.shape[0] > seq_emb.shape[0]:
            print(seq_level_pos)
            print(item_level_pos)
            print(noise_score)
            print(noise_prob)
            print(noise_pos.shape)
            print(len(noise_pos.shape))
            assert 1 == 2
        seq_emb_1st, seq_len_1st, de_flag_1st = self._denoise_rule(noise_pos, seq_emb, seq_len, seq_emb_1st_tmp,
                                                                   seq_len_1st_temp)

        # seq_emb_1st, seq_len_1st, de_flag_1st = self._denoise_rule(noise_prob, seq_emb, seq_len, seq_emb_1st_tmp,
        #                                                            seq_len_1st_temp)
        # return: batch x item_num    ;   batch x item_num x dim  ; batch  ; batch
        return noise_prob.squeeze(), seq_emb_1st, seq_len_1st, de_flag_1st

    def forward_2nd(self, noise_prob_1st, de_flag_1st, seq_emb, seq_len, masks, tau, item_seq=0, hard=True):

        mask, mask_row, mask_col = masks
        # seq_emb_pos, seq_emb_neg = seq_embs
        seq_and_mask_pos = (seq_emb, mask, mask_row, mask_col)

        seq_level_pos = self._seq_consistency(seq_and_mask_pos)
        item_level_pos = self._item_consistency(seq_and_mask_pos)
        # noise_score = seq_level * item_level
        noise_score = 1 - (item_level_pos * seq_level_pos)

        # all position denoising
        noise_prob_2nd = F.gumbel_softmax(noise_score, tau=tau, hard=hard).unsqueeze(-1)  # batch x item_num x 1
        noise_prob = self._denoise_rule_2nd(noise_prob_1st, noise_prob_2nd, de_flag_1st).unsqueeze(
            -1)  # batch x item_num x 1
        seq_emb_2nd = seq_emb   # batch x item_num x dim
        seq_len_2nd = seq_len - torch.sum(noise_prob.squeeze(), 1)  # batch
        de_flag_2nd = seq_len - seq_len_2nd
        # input: batch x item_num x 1, batch x item_num x dim, batch, batch x item_num x dim, batch

        return noise_prob_2nd.squeeze(), seq_emb_2nd, seq_len_2nd.to(
            seq_len.dtype), de_flag_2nd, item_seq * (1 - noise_prob).squeeze()

    def _get_cat_seq(self, item_seq_emb):
        start = self.se_embedding.weight[0]
        end = self.se_embedding.weight[1]
        one = torch.ones_like(item_seq_emb[:, 0, :]).unsqueeze(1)
        cat_item_seq_emb = torch.cat([start * one, item_seq_emb, end * one], dim=1)
        # return : batch x item_num x dim
        return cat_item_seq_emb

    def _seq_consistency(self, seq_and_mask):
        item_seq_emb, mask, mask_row, mask_col = seq_and_mask
        att_mask = (mask - 1) * 2e32
        cat_item_seq_emb = self._get_cat_seq(item_seq_emb)
        hn, _ = self.Bi_LSTM(cat_item_seq_emb)
        hn_fw = hn[:, :-2, :self.hidden_size]  # f1, f2, ..., fn  -> batch x item_num x dim
        hn_bw = hn[:, 2:, self.hidden_size:]  # b1, b2, ..., bn  -> batch x item_num x dim
        consistency = torch.sum(item_seq_emb * (hn_fw * hn_bw), dim=-1) + att_mask  # batch x item_num  # 加改成×
        consistency_prob = self.softmax(consistency)
        return consistency_prob

    def _item_consistency(self, seq_and_mask):
        # batch x len x dim
        item_seq_emb, mask, mask_row, mask_col = seq_and_mask
        att_mask = (mask - 1) * 2e32
        clamp_emb = torch.clamp(item_seq_emb,min=-1.0, max=1.0)
        # sim = item_seq_emb @ item_seq_emb.transpose(2, 1)  # batch x len x len

        sim = clamp_emb @ clamp_emb.transpose(2, 1)  # batch x len x len
        diag_self = torch.diagonal(sim, dim1=-1, dim2=-2)
        sim_self = torch.diag_embed(diag_self)
        sim_res = (sim * mask_row * mask_col - sim_self).squeeze()
        consistency = torch.sum(sim_res, dim=1) + att_mask
        consistency_prob = self.softmax(consistency)
        # print('*********************')
        # print('consistency',consistency)
        # print(consistency_prob,consistency_prob)
        # print('*********************')
        return consistency_prob

        # 'Non - Implmentation'
        # return 0

    # def _denoise_rule(self, noise_score, ori_seq, ori_len, de_seq, de_len):
    def _denoise_rule(self, noi_pos, ori_seq, ori_len, de_seq, de_len):
        mask_len = (noi_pos < ori_len).type(ori_len.dtype)
        mask_emb = mask_len.unsqueeze(1).unsqueeze(1)
        de_seq = mask_emb * de_seq + (1 - mask_emb) * ori_seq  # batch x item_num x dim
        de_len = mask_len * de_len + (1 - mask_len) * ori_len
        de_flag = ori_len - de_len  # 1: denoise, 0: not denoise (batch)
        return de_seq, de_len, de_flag

    def _denoise_rule_2nd(self, noise_prob1, noise_prob2, de_flag):
        # np1 = [0,1,0,0,0]   np2 = [0,0,1,0,0]
        r1 = torch.roll(noise_prob1, 1, 1)  # [0,0,1,0,0]
        r2 = torch.roll(r1, 1, 1)  # [0,0,0,1,0]
        noise_prob_2 = (noise_prob1 + r1 + r2) * noise_prob2.squeeze() * de_flag.unsqueeze(
            1)  # [0,1,1,1,0] x [0,0,1,0,0] = [0,0,1,0,0]
        # return: batch x item_num
        return noise_prob_2
