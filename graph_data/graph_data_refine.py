import torch
import numpy as np
from scipy import sparse
import pickle


def dense2sparse(_matrix,device):
    a_ = sparse.coo_matrix(_matrix)
    v1 = a_.data
    indices = np.vstack((a_.row, a_.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(v1)
    shape = a_.shape
    if torch.cuda.is_available():
        sparse_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(torch.float32).to(device)
    else:
        sparse_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_matrix

if __name__ == '__main__':
    # datas =  ['amazon-sports-outdoors','yelp']
    # datas =  ['ml-100k']
    datas = ['amazon-beauty']
    for data in datas:
        print('refine data of:',data)
        file = data + '/graph_data.pkl'
        with open(file, 'rb') as f:
            graphs = pickle.load(f)
            # g_user_item, g_user_item_t,_ = graphs
        print('refine data')
        g_user_item,g_user_item_t,g3,g4,g5,g6,g7 = graphs
        # print(type(g_user_item))
        # print(type(g_user_item_t))
        a = g_user_item.to_dense().data.cpu().numpy()
        # print(a[:5,:5])
        # assert 1==2
        at = g_user_item_t.to_dense().data.cpu().numpy()

        a_u_sum = np.sum(a, 1)
        a_u_sum_mask = 1 - (a_u_sum > 0.01) * 1
        a_u_sum_mask = np.reshape(a_u_sum_mask, (a.shape[0], 1))
        a_u_sum = np.reshape(a_u_sum, (a.shape[0], 1)) + a_u_sum_mask

        a_i_sum = np.sum(at, 1)
        a_i_sum_mask = 1 - (a_i_sum > 0.01) * 1
        a_i_sum_mask = np.reshape(a_i_sum_mask, (a.shape[1], 1))
        a_i_sum = np.reshape(a_i_sum, (a.shape[1], 1)) + a_i_sum_mask

        a_u = a / a_u_sum
        a_i = at / a_i_sum
        g_u = dense2sparse(a_u,g3.device)
        g_u_t = dense2sparse(a_i,g3.device)
        new_graphs = (g_u,g_u_t,g3,g4,g5,g6,g7)

        with open(file, 'wb') as f:
            pickle.dump(new_graphs, f)




        # print(type(g_user_item_t))