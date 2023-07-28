import math

from tqdm import tqdm
import torch
import numpy as np
import scipy.sparse as sp
from scipy import sparse
import pickle
# def create_user_sim_graph(seqs):
import argparse


class GraphData(object):
    def __init__(self, config, data, dataset_name='demo', is_demo=False):
        self.less_freq = 10
        self.cold_ratio = 0.8
        self.device = config['device']
        self.data = data
        self.is_demo = is_demo
        self.dataset_name = dataset_name
        if is_demo:
            self.seq_data = self._get_data_demo()
            self.graph_data_file = dataset_name + '/graph_data.pkl'
            self.graph_info_file = dataset_name + '/graph_info.pkl'
        else:
            self.seq_data = self._get_data()
            self.graph_data_file = 'graph_data/' + dataset_name + '/graph_data.pkl'
            self.graph_info_file = 'graph_data/' + dataset_name + '/graph_info.pkl'

        self.is_refined = False
        # load graph data
        try:
            print('load graphs')
            with open(self.graph_data_file, 'rb') as f:
                graphs = pickle.load(f)
                a_u,a_i,g3,g4,g5,g6,g7 = graphs
                self.refine_ui = True
                g_user_item, g_user_item_t = self._get_graphs()
                gs = g_user_item, g_user_item_t,g3,g4,g5,g6,g7
            print('refine graphs')
            with open(self.graph_data_file, 'wb') as f:
                pickle.dump(gs, f)
            self.is_refined = True

        except:
            assert 1==2

    def _get_graphs(self):
        if self.refine_ui:
            g_user_item, g_user_item_t = self._multi_graph2numpy()
            return self._dense2sparse(g_user_item), self._dense2sparse(g_user_item_t)
        g_user_item, g_user_item_t, graph_pos_item, graph_neg_item, graph_pos_user, graph_neg_user = self._multi_graph2numpy()
        # g_user_item, _ = self._get_degree_matrix(graph_user_item)
        g_pos_item_in, g_pos_item_out = self._get_degree_matrix(graph_pos_item)
        g_neg_item, _ = self._get_degree_matrix(graph_neg_item)
        g_pos_user, _ = self._get_degree_matrix(graph_pos_user)
        g_neg_user, _ = self._get_degree_matrix(graph_neg_user)

        g_user_item = self._dense2sparse(g_user_item)
        g_user_item_t = self._dense2sparse(g_user_item_t)

        degree_graphs = g_user_item, g_user_item_t, g_pos_item_in, g_pos_item_out, g_neg_item, g_pos_user, g_neg_user
        if self.is_demo:
            self._demo_print(degree_graphs, type='degree')
        return degree_graphs

    def _get_data(self):
        # tr = ([v1], v2)
        # va = ([v1, v2], v3)
        # te = ([v1, v2, v3], v4) = (seq, next-item)
        dataset = self.data.dataset

        data = dataset.inter_feat.interaction
        user_num, item_num = dataset.user_num, dataset.item_num  # item_num=1350 / max_item=1349
        # 构造超图，因为一个用户增强出来很多，所以不如是用用户做超边
        user = data['user_id']  # tensor: seq_num ,  USER_ID = 1:user_num-1
        seq = data['item_id_list']  # tensor: seq_num x seq_len
        seq_len = data['item_length']  # tensor: seq_num
        # print('user_num',user_num)
        # print('user_len',len(user))
        # print('min_user',torch.min(user))
        # print('max_user',torch.max(user))
        # print('user:943',user[942])
        return user_num, item_num, user, seq, seq_len

    def _get_base_graphs(self):
        print('get base graphs')
        user_num, item_num, users, seqs, seq_lens = self.seq_data
        seqs_list = seqs.tolist()
        lens_list = seq_lens.tolist()
        seqs_set = [set(seq) for seq in seqs_list]

        graph_base_item = torch.zeros((item_num, item_num))
        graph_base_user = torch.zeros((user_num, user_num))
        graph_pos_item = torch.zeros((item_num, item_num))
        graph_base_ui = torch.zeros((user_num, item_num))
        user_counts = lens_list
        item_counts = [0] * item_num
        # create user-item graph
        for u_id in tqdm(range(user_num - 1), total=user_num - 1, ncols=100):  # user_num = 0:943  user_u = 1:943
            user = users[u_id]
            seq = seqs[u_id]
            for item_i in seq:
                graph_base_ui[user,item_i] +=1
        if self.refine_ui:
            return graph_base_ui
        # create item base&pos graph
        for u_id in tqdm(range(user_num - 1), total=user_num - 1, ncols=100):  # user_num = 0:943  user_u = 1:943
            seq = seqs_list[u_id]
            len = lens_list[u_id]
            id_range = range(len)
            item_counts[seq[len-1]] += 1  # 倒数第一个item直接次数+1
            for i_id in id_range[:-1]:  # 一直扫描到序列倒数第二个item
                item_i = seq[i_id]
                item_counts[item_i] += 1  # item_counts + 1
                # print(id_range[i_id+1])
                for j_id in id_range[i_id + 1:]:
                    item_j = seq[j_id]
                    dis = j_id - i_id
                    weight = (len - dis) / len
                    graph_base_item[item_i, item_j] += 1
                    graph_pos_item[item_i, item_j] += weight

        # create user base graph
        # 有一套优化方案是根据user u交互过的每个item去找共同交互过的user v，那么共同交互的总数就加到相应的u-v相似度上
        for u_id in tqdm(range(user_num - 1), total=user_num - 1, ncols=100):  # user_num = 0:943  user_u = 1:943
            user_u = users[u_id]
            seq_u = seqs_list[u_id]
            set_u = seqs_set[u_id]
            len_u = lens_list[u_id]
            item_i_list = seq_u[:len_u]

            for v_id in range(user_num - 1)[u_id + 1:]:
                user_v = users[v_id]
                seq_v = seqs_list[v_id]
                set_v = seqs_set[v_id]
                len_v = lens_list[v_id]
                com_item_num = 0
                total_item_num = len_u + len_v
                item_j_list = seq_v[:len_v]

                for item_i in item_i_list:
                    if item_i in set_v:
                        com_item_num += 1
                for item_j in item_j_list:
                    if item_j in set_u:
                        com_item_num += 1
                sim_u_v = com_item_num / total_item_num
                graph_base_user[user_u, user_v] = sim_u_v

        return graph_base_ui, graph_base_item, graph_base_user, graph_pos_item, item_counts, user_counts

    def _get_base_graphs1(self):
        print('get base graphs')
        user_num, item_num, users, seqs, seq_lens = self.seq_data
        graph_base_item = torch.zeros((item_num, item_num))
        graph_base_user = torch.zeros((user_num, user_num))
        graph_pos_item = torch.zeros((item_num, item_num))
        graph_base_ui = torch.zeros((user_num, item_num))
        user_counts = [0] * user_num
        item_counts = [0] * item_num
        for u_id in tqdm(range(user_num - 1), total=user_num - 1, ncols=100):  # user_num = 0:943  user_u = 1:943
            user_u = users[u_id]  # user id
            seq_u = seqs[u_id]  # sequence
            len_seq_u = seq_lens[u_id]  # sequence length
            user_counts[user_u] = len_seq_u
            count_user = 0
            for i_id in range(seq_u.shape[0]):  # max_seq_len
                item_i = seq_u[i_id]
                if item_i != 0:
                    count_user += 1  # count_user + 1
                    item_counts[item_i] += 1  # item_counts + 1
                    graph_base_ui[user_u, item_i] += 1  # U-I Graph += 1
                    for j_id in range(seq_u.shape[0])[i_id + 1:]:  # i->j next-item global_graph+1
                        item_j = seq_u[j_id]
                        if item_j != 0:
                            dis = j_id - i_id
                            weight = (len_seq_u - dis) / len_seq_u
                            graph_base_item[item_i, item_j] += 1
                            graph_pos_item[item_i, item_j] += weight
                        else:
                            continue
                continue

            for v_id in range(user_num - 1)[u_id + 1:]:
                user_v = users[v_id]
                seq_v = seqs[v_id]
                len_seq_v = seq_lens[v_id]
                com_item_num = 0
                for i_id in range(seq_u.shape[0]):
                    item_i = seq_u[i_id]
                    if item_i == 0:
                        continue
                    if item_i in seq_v:
                        com_item_num += 1
                for j_id in range(seq_v.shape[0]):
                    item_j = seq_v[j_id]
                    if item_j == 0:
                        continue
                    if item_j in seq_u:
                        com_item_num += 1
                total_item_num = len_seq_u + len_seq_v
                sim_u_v = com_item_num / total_item_num
                graph_base_user[user_u, user_v] = sim_u_v
        return graph_base_ui, graph_base_item, graph_base_user, graph_pos_item, item_counts, user_counts

    def _get_multi_graphs(self):
        user_num, item_num, users, seqs, seq_lens = self.seq_data
        if self.refine_ui:
            graph_base_ui = self._get_base_graphs()
            graph_user_item = graph_base_ui
            graph_user_item_t = graph_user_item.t()
            return graph_user_item, graph_user_item_t
        graph_base_ui, graph_base_item, graph_base_user, graph_pos_item, item_counts, user_counts = self._get_base_graphs()

        # graph_neg_user = np.zeros((user_num, user_num))  # user1:浪漫+爱情；user2:浪漫+科幻; user3:科幻+警匪
        # graph_neg_item = np.zeros((item_num, item_num))

        item_freq = torch.tensor(item_counts[1:])
        item_freq_sorted = torch.sort(item_freq, descending=False)[1] + 1  # 映射到真实的item id
        cold_item = item_freq_sorted[:math.ceil(len(item_freq) * self.cold_ratio)]

        user_freq = torch.tensor(user_counts[1:])
        user_freq_sorted = torch.sort(user_freq, descending=False)[1] + 1 # 映射到真实的 user id
        cold_user = user_freq_sorted[:math.ceil(len(user_freq_sorted) * (self.cold_ratio+0.1))]
        print('get item graphs')
        graph_pos_item = graph_pos_item * graph_base_item.gt(self.less_freq)
        graph_neg_item = self._get_graph_neg_item(graph_base_item, cold_item, self.less_freq)

        print('get user graphs')
        graph_pos_user = graph_base_user + graph_base_user.T + torch.diag(torch.ones(user_num))

        print('graph pos user', graph_pos_user)
        graph_neg_user = self._get_graph_neg_user(graph_pos_user, cold_user)

        graph_user_item = graph_base_ui
        graph_user_item_t = graph_user_item.t()
        graphs = graph_user_item, graph_user_item_t, graph_base_item, graph_pos_item, graph_neg_item, graph_pos_user, graph_neg_user

        graphs_info = self._get_graph_info(graphs)

        with open(self.graph_info_file, 'wb') as f:
            pickle.dump(graphs_info, f)

        if self.is_demo:
            self._demo_print(graphs)
            print(graphs_info)

        return graphs

    def _multi_graph2numpy(self):
        if self.refine_ui:
            graph_user_item, graph_user_item_t = self._get_multi_graphs()
            return graph_user_item.numpy(), graph_user_item_t.numpy()
        graph_user_item, graph_user_item_t, graph_base_item, graph_pos_item, graph_neg_item, graph_pos_user, graph_neg_user = self._get_multi_graphs()
        return graph_user_item.numpy(), graph_user_item_t.numpy(), graph_pos_item.numpy(), graph_neg_item.numpy(), graph_pos_user.numpy(), graph_neg_user.numpy()

    def _get_graph_neg_item(self, g, cold_item_, less_freq_=100):
        # tensor, list, int
        # weight = \sum_{v \in N_v} count(v_i-v)+count(v_j-v)
        graph_ = g + g.T
        never_co_exist_ = graph_.eq(0)
        graph_ = graph_ - torch.diag(torch.diag(graph_))
        gt_less_freq_ = graph_.gt(less_freq_)
        graph_ = graph_ * gt_less_freq_  # 断掉所有交互次数过少的边

        two_hop_ = graph_ @ graph_
        same_context_ = two_hop_.gt(0)

        is_incompatible_ = never_co_exist_ * same_context_

        # is_incompatible_ = ~is_1_hop_ * is_2_hop_ * ~is_3_hop_
        in_graph_ = 1 * is_incompatible_
        in_graph_ = in_graph_ - torch.diag(torch.diag(in_graph_))  # get in_graph (0,1)matrix

        '不考虑long-tail item的互斥性'
        in_graph_[cold_item_, :] = 0
        in_graph_[:, cold_item_] = 0

        incompatible_strength_graph_ = torch.zeros(graph_.shape)
        for node1 in tqdm(range(graph_.shape[0]),total=graph_.shape[0],ncols=100):  # item a

            # for node1 in tqdm(range(graph_.shape[0]), total=graph_.shape[0], ncols=100):  # item a
            node2 = torch.squeeze(torch.nonzero(in_graph_[node1]))  # incompatible item b

            if node2.shape == torch.Size([0]) or node2 == torch.Size([]):
                continue

            node1_neighbor_freq_ = graph_[node1]
            node2_neighbor_freq_ = graph_[node2]

            node1_neighbor_ = node1_neighbor_freq_.gt(0)
            node2_neighbor_ = node2_neighbor_freq_.gt(0)

            is_common_ = node1_neighbor_ * node2_neighbor_  # bridges between item a and b

            if len(is_common_.shape) == 1:
                node1_common_freq_ = torch.sum(is_common_ * node1_neighbor_freq_)
                node2_common_freq = torch.sum(is_common_ * node2_neighbor_freq_)
            else:
                node1_common_freq_ = torch.sum(is_common_ * node1_neighbor_freq_, dim=1)
                node2_common_freq = torch.sum(is_common_ * node2_neighbor_freq_, dim=1)

            in_strength_ = node1_common_freq_ + node2_common_freq

            incompatible_strength_graph_[node1, node2] = in_strength_

        in_graph_ = in_graph_ * incompatible_strength_graph_  # get incompatible graph with in-strength

        is_undirected_ = torch.eq(in_graph_, in_graph_.T)

        assert is_undirected_.sum() == graph_.shape[0] * graph_.shape[1]

        # return tensor (float)
        return in_graph_

    def _get_graph_neg_user(self, g, cold_user):
        print('user base graph',g)
        # tensor, list, int
        # weight = \sum_{v \in N_v} count(v_i-v)+count(v_j-v)
        graph_ = g
        never_co_exist_ = graph_.eq(0)
        graph_ = graph_ - torch.diag(torch.diag(graph_))
        # gt_less_sim_ = graph_.gt(less_sim_)
        # graph_ = graph_ * gt_less_sim_  # 断掉所有相似度过低的边

        two_hop_ = graph_ @ graph_
        same_context_ = two_hop_.gt(0)

        is_incompatible_ = never_co_exist_ * same_context_
        print('never co exist',never_co_exist_.shape)
        print('same context', same_context_.shape)
        print(is_incompatible_.shape)

        # is_incompatible_ = ~is_1_hop_ * is_2_hop_ * ~is_3_hop_
        in_graph_ = 1 * is_incompatible_
        in_graph_ = in_graph_ - torch.diag(torch.diag(in_graph_))  # get in_graph (0,1)matrix

        '不考虑padding user'
        print(in_graph_.shape)
        in_graph_[0, :] = 0
        in_graph_[:, 0] = 0
        in_graph_[cold_user, :] = 0
        in_graph_[:, cold_user] = 0
        cold_user_set = set(cold_user)
        incompatible_strength_graph_ = torch.zeros(graph_.shape)
        for node1 in tqdm(range(graph_.shape[0]),total=graph_.shape[0],ncols=100):  # item a
            if node1 in cold_user_set:
                continue
            # for node1 in tqdm(range(graph_.shape[0]), total=graph_.shape[0], ncols=100):  # item a
            node2 = torch.squeeze(torch.nonzero(in_graph_[node1]))  # incompatible item b

            if node2.shape == torch.Size([0]) or node2 == torch.Size([]):
                continue

            node1_neighbor_freq_ = graph_[node1]
            node2_neighbor_freq_ = graph_[node2]

            node1_neighbor_ = node1_neighbor_freq_.gt(0)
            node2_neighbor_ = node2_neighbor_freq_.gt(0)

            is_common_ = node1_neighbor_ * node2_neighbor_  # bridges between item a and b

            if len(is_common_.shape) == 1:
                node1_common_freq_ = torch.sum(is_common_ * node1_neighbor_freq_)
                node2_common_freq = torch.sum(is_common_ * node2_neighbor_freq_)
            else:
                node1_common_freq_ = torch.sum(is_common_ * node1_neighbor_freq_, dim=1)
                node2_common_freq = torch.sum(is_common_ * node2_neighbor_freq_, dim=1)

            in_strength_ = node1_common_freq_ + node2_common_freq

            incompatible_strength_graph_[node1, node2] = in_strength_

        in_graph_ = in_graph_ * incompatible_strength_graph_  # get incompatible graph with in-strength

        is_undirected_ = torch.eq(in_graph_, in_graph_.T)

        assert is_undirected_.sum() == graph_.shape[0] * graph_.shape[1]

        # return tensor (float)
        return in_graph_

    def _get_degree_matrix(self, adj_matrix):
        '''
        A = [ 1, 2, 2,
              0, 4, 6,
              1, 0, 0 ]
        in = [ 0.5, 0.0, 0.5,
               0.3,  0.7,  0.0,
               1.0,  0.0,  0.0 ]
        out = [ 0.2, 0.4, 0.4,
                0.0  0.4  0.6,
                1.0  0.0  0.0 ]
        NOTE: E = AE --> E \in R^{n X d}
        '''

        d = np.shape(adj_matrix)[0]
        row_temp = np.sum(adj_matrix, axis=0)
        row = self._bool_numpy(row_temp)
        row = np.reshape(row, (1, d))
        col_temp = np.sum(adj_matrix, axis=1)
        col = self._bool_numpy(col_temp)
        col = np.reshape(col, (d, 1))
        a_out = adj_matrix / col
        a_in = adj_matrix / row
        a_in = a_in.T

        a_in = self._dense2sparse(a_in)
        a_out = self._dense2sparse(a_out)

        # a_out = torch.from_numpy(a_out)
        # a_in = torch.from_numpy(a_in)
        return a_in, a_out

    def _dense2sparse(self, _matrix):
        a_ = sparse.coo_matrix(_matrix)
        v1 = a_.data
        indices = np.vstack((a_.row, a_.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(v1)
        shape = a_.shape
        if torch.cuda.is_available():
            sparse_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(torch.float32).to(self.device)
        else:
            sparse_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        return sparse_matrix

    def _bool_numpy(self, numpy_array):
        numpy_array_1 = numpy_array.copy()
        numpy_array_1[numpy_array_1 == 0.] = 1
        return numpy_array_1

    def _graph_info(self, g):
        shape = g.shape[0] * g.shape[1]
        edges = torch.sum(g.gt(0))
        sparsity = edges / shape
        return {'shape': shape, 'edges': edges, 'sparsity': sparsity}

    def _get_graph_info(self, graphs):
        graph_info = {}
        graph_user_item, graph_user_item_t, graph_base_item, graph_pos_item, graph_neg_item, graph_pos_user, graph_neg_user = graphs
        graph_info['user item'] = self._graph_info(graph_user_item)
        graph_info['item base'] = self._graph_info(graph_base_item)
        graph_info['item pos'] = self._graph_info(graph_pos_item)
        graph_info['item neg'] = self._graph_info(graph_neg_item)
        graph_info['user pos'] = self._graph_info(graph_pos_user)
        graph_info['user neg'] = self._graph_info(graph_neg_user)

        return graph_info

    def _get_data_demo(self):
        user_num = 4
        item_num = 6
        user = torch.tensor([1, 2, 3])
        seq = torch.tensor([[1, 2, 3, 4],
                            [1, 2, 0, 0],
                            [3, 4, 5, 0]])
        seq_len = torch.tensor([4, 2, 3])

        # user-item graph
        #   = [[0, 1, 1, 1, 1, 0]
        #      [2, 1, 1, 0, 0, 0],
        #      [1, 0, 0, 1, 1, 1]]

        # item-base graph
        #   = [[0, 0, 0, 0, 0, 0],
        #      [0, 0, 2, 1, 1, 0],
        #      [0, 0, 0, 1, 1, 0],
        #      [0, 0, 0, 0, 2, 1],
        #      [0, 0, 0, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 0]]

        # item-pos graph
        #   = [[0, 0, 0, 0, 0, 0],                           [0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
        #      [0, 0, 3/4+1/2, 2/4, 1/4, 0],                 [0.00, 0.00, 1.25, 0.50, 0.25, 0.00]
        #      [0, 0, 0, 3/4, 2/4, 0],                       [0.00, 0.00, 0.00, 0.75, 0.50, 0.00]
        #      [0, 0, 0, 0, 6/4, 2/4],                       [0.00, 0.00, 0.00, 0.00, 1.41, 0.33]
        #      [0, 0, 0, 0, 0, 0],                           [0.00, 0.00, 0.00, 0.00, 0.00, 0.66]
        #      [0, 0, 0, 0, 0, 0]]                           [0.00, 0.00, 0.00, 0.00, 0.00, 0.00]

        # item-neg graph
        #   = [[0, 0, 0, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 4],
        #      [0, 0, 0, 0, 0, 4],
        #      [0, 0, 0, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 0],
        #      [0, 4, 4, 0, 0, 0]]

        # user-pos graph
        #   = [[0, 4/6, 4/7],            [1.00, 0.66, 0.57]
        #      [4/6, 0, 0],              [0.66, 1.00, 0.00]
        #      [4/7, 0, 0],              [0.57, 0.00, 1.00]

        # user-neg graph
        #   = [[0, 0, 0],
        #      [0, 0, 4/6+4/7],
        #      [0, 4/6+4/7, 0],

        # user-pos graph
        #   = [[0, 0, 0, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 4],
        #      [0, 0, 0, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 0],
        #      [0, 4, 0, 0, 0, 0]]

        return user_num, item_num, user, seq, seq_len

    def _demo_print(self, graphs, type='graphs'):
        if type == 'graphs':
            graph_user_item, g_user_item_t, graph_base_item, graph_pos_item, graph_neg_item, graph_pos_user, graph_neg_user = graphs
            print('user-item', graph_user_item)
            print('item base', graph_base_item)
            print('item pos', graph_pos_item)
            print('item neg', graph_neg_item)
            print('user pos', graph_pos_user)
            print('user neg', graph_neg_user)
        elif type == 'degree':
            g_user_item, g_user_item_t, g_pos_item_in, g_pos_item_out, g_neg_item, g_pos_user, g_neg_user = graphs
            print('degree user item:', g_user_item)
            print('degree pos item in:', g_pos_item_in)
            print('degree pos item out:', g_pos_item_out)
            print('degree neg item:', g_neg_item)
            print('degree pos user:', g_pos_user)
            print('degree neg user:', g_neg_user)


if __name__ == '__main__':
    config_demo = {'less_freq': 0, 'cold_ratio': 0, 'device': 'cuda:0'}
    data_demo = 0
    graph_data = GraphData(config_demo, data_demo, is_demo=True)
    graphs = graph_data.graphs
    graphs_info = graph_data.graphs_info
    print(graphs_info)
    pass
