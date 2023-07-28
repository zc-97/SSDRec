from config.base_init import pre_train_model_dict
import openpyxl
import yaml
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.config import Config
from dataset.dataUtils import create_dataset, data_preparation
from model.hsd import HSD
from graph_data.data_process0706 import GraphData
# from graph_data.re_create_graph_ui import GraphData
from Triainer.HsdTrainer import HsdTrainer
from HsdUtils.utils import get_model, is_hsd_model

lr_ = 1e-3
data_augmentation = True

gumbel_temperature = 0.5
is_gumbel_tau_anneal = True
long_tail_rate = 0.2
weight_decay = 0
seed = 42

our_ae_drop_out = 0.3
our_att_drop_out = 0.2
load_pre_train_emb = False

gpu_id = 0
embedding_size = 100


def main(model_name, dataset, sub_model):
    data_name = dataset
    note = 'hyper-parameter analysis, embedding size:[%d]' % embedding_size
    config_file_list = ['config/text.yml']

    parameter_dict = get_parameter_dict(model_name, dataset, sub_model)

    print('!!!Note: ' + note)

    config = Config(
        model=HSD if is_hsd_model(model_name) else model_name,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=parameter_dict
    )
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)


    if mode=='refine' and config['is_dna'] and 'HSD' in model_name:
        g = GraphData(config, test_data, data_name)
        is_refined = g.is_refined
        return is_refined
    if config['is_dna'] and 'HSD' in model_name:
        g = GraphData(config, test_data, data_name)
        graphs = g.graphs
        graphs_info = g.graphs_info
        print(graphs_info)
        config['graphs'] = graphs
    # model loading and initialization
    model_class = get_model(model_name)
    model = model_class(config, train_data.dataset).to(config['device'])
    # print(config['device'])
    # de = str(config['device']) + ':' + str(gpu_id)
    # model = model_class(config, train_data.dataset).to(de)
    logger.info(model)

    # trainer loading and initialization
    trainer = HsdTrainer(config, model)
    # trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))

    rst_dic = {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
    
    record_model_result(data_name, model_name, parameter_dict, rst_dic, note, config_file_list, trainer.saved_model_file)


def get_parameter_dict(model_name, dataset, sub_model):
    parameter_dict = {
        "gpu_id": gpu_id,
        "seed": seed,
        "weight_decay": weight_decay,

        'loss_type': 'BPR',
        'sub_model': sub_model,

        # bert4rec 最大序列长度200 for ml-1m，50 for others
        "MAX_ITEM_LIST_LENGTH": 200 if dataset == 'ml-1m' else 50,
        "data_augmentation": data_augmentation,
        "user_inter_num_interval": '(4, inf)',
        "item_inter_num_interval": '(4, inf)',
        "initializer_range": 0.02,
        "learning_rate": lr_,
        "scheduler": False,
        "step_size": 4,
        "gamma": 0.1,

        "load_col": {'inter': ['user_id', 'item_id', 'rating', 'timestamp']} if dataset != 'yelp1' else {
            'inter': ['user_id', 'business_id', 'stars', 'date']},
        "USER_ID_FIELD": 'user_id',
        "ITEM_ID_FIELD": 'item_id' if dataset != 'yelp1' else 'business_id',
        "RATING_FIELD": 'rating' if dataset != 'yelp1' else 'stars',
        "TIME_FIELD": 'timestamp' if dataset != 'yelp1' else 'date',

        "our_att_drop_out": our_att_drop_out,
        "our_ae_drop_out": our_ae_drop_out,

        "gumbel_temperature": gumbel_temperature,
        "is_gumbel_tau_anneal": is_gumbel_tau_anneal,
        "long_tail_rate": 0.95,
        'neg_sampling': {
            'uniform': 1
        },
        "pre_train_model_dict": pre_train_model_dict,
        "load_pre_train_emb": load_pre_train_emb

    }
    if dataset == 'yelp':
        # parameter_dict['val_interval'] = {'date': '[1546272000, inf]'}
        parameter_dict['val_interval'] = {'timestamp': '[1546272000, inf]'}
    if model_name == 'BERT4Rec' or (is_hsd_model(model_name) and sub_model == 'BERT4Rec'):
        bert_dict = {
            "embedding_size": embedding_size,
            "wandb_project": "recbole",
            "require_pow": False,
            "n_layers": 2,
            "n_heads": 2,
            "hidden_size": embedding_size,
            "inner_size": embedding_size,
            "hidden_dropout_prob": 0.5,
            "attn_dropout_prob": 0.5,
            "hidden_act": 'gelu',
            "layer_norm_eps": 1e-12,
            "mask_ratio": 0.6
        }
        parameter_dict.update(bert_dict)
    elif model_name == 'GRU4Rec' or (is_hsd_model(model_name) and sub_model == 'GRU4Rec'):
        gru_dict = {
            # "embedding_size": 64,
            "embedding_size": embedding_size,
            "hidden_size": 100,
            "num_layers": 1,
            "dropout_prob": 0.3
        }
        parameter_dict.update(gru_dict)

    elif model_name == 'SASRec' or (is_hsd_model(model_name) and sub_model == 'SASRec'):
        sas_dict = {
            "n_layers": 2,
            "n_heads": 2,
            "hidden_size": embedding_size,
            "inner_size": 100,
            "hidden_dropout_prob": 0.5,
            "attn_dropout_prob": 0.5,
            "hidden_act": 'gelu',
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
            "loss_type": 'BPR',
            'neg_sampling': {
                'uniform': 5
            }
        }
        parameter_dict.update(sas_dict)

    elif model_name == 'Caser' or (is_hsd_model(model_name) and sub_model == 'Caser'):
        caser_dict = {
            "embedding_size": embedding_size,
            "hidden_size": 100,
            "dropout_prob": 0.4,
            "reg_weight": 1e-4,
            "weight_decay": 0.0,
            "nv": 8,
            "nh": 16,
            # "loss_type": 'CE',
            "MAX_ITEM_LIST_LENGTH": 5,
        }
        parameter_dict.update(caser_dict)

    elif model_name == 'NARM' or (is_hsd_model(model_name) and sub_model == 'NARM'):
        narm_dict = {
            "embedding_size": embedding_size,
            "hidden_size": 100,
            "n_layers": 1,
            "dropout_probs": [0.25, 0.5],
            # "loss_type": 'CE'
        }
        parameter_dict.update(narm_dict)

    elif model_name == 'GCSAN' or (is_hsd_model(model_name) and sub_model == 'GCSAN'):
        GCSAN_dict = {
            "n_layers": 1,
            "n_heads": 1,
            "hidden_size": embedding_size,
            "inner_size": 100,
            "hidden_dropout_prob": 0.2,
            "attn_dropout_prob": 0.2,
            "hidden_act": 'gelu',
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
            "step": 1,
            "weight": 0.6,
            "reg_weight": 5e-5,
            "weight_decay": 0.0,
            # "loss_type": 'CE'
        }
        parameter_dict.update(GCSAN_dict)

    elif model_name == 'SRGNN' or (is_hsd_model(model_name) and sub_model == 'SRGNN'):
        SRGNN_dict = {
            "embedding_size": embedding_size,
            "hidden_size": 100,
            "step": 1,
            # "loss_type": 'CE'
        }
        parameter_dict.update(SRGNN_dict)

    elif model_name == 'fmlp' or (is_hsd_model(model_name) and sub_model == 'fmlp'):
        FMLP_dict = {
            "hidden_size": embedding_size,
            "hidden_dropout_prob": 0.5,
            "initializer_range": 0.02,
            # "loss_type": 'CE',
        }
        parameter_dict.update(FMLP_dict)

    elif model_name == 'DSAN' or (is_hsd_model(model_name) and sub_model == 'DSAN'):
        dsan_dict = {
            "hidden_size": embedding_size,
            "item_dim1": 100,
            "pos_dim1": 100,
            "dim1": 100,
            "weight_decay": 1e-3,
            "amsgrad": True,
            "w": 10,
            # "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
            # "loss_type": 'CE'
        }
        parameter_dict.update(dsan_dict)

    elif model_name == 'STAMP' or (is_hsd_model(model_name) and sub_model == 'STAMP'):
        dsan_dict = {
            "embedding_size": embedding_size,
            "hidden_size": 100,
            # "loss_type": 'CE'
        }
        parameter_dict.update(dsan_dict)
    return parameter_dict


def record_model_result(dataset, model_name, parameter_dict, rst_dic, note, config, saved_model_file):
    config = _load_config_files(config)
    wb = openpyxl.load_workbook(output_file_name)
    if config['is_dna']:
        is_dna = '+dna'
    else:
        is_dna = ''
    if 'HSD' in model_name:
        model_name += is_dna + '_' + parameter_dict['sub_model']
    eval_model = config['eval_args']['mode']
    dataset += ('/' + eval_model) if eval_model is not None else ''
    run_rst = [dataset, model_name]
    test_rst = list(rst_dic['test_result'].values())
    run_rst.extend(test_rst)
    run_rst.append('\t')
    run_rst.append(str(parameter_dict))
    run_rst.append(note)
    run_rst.append(saved_model_file)

    sheet = wb['Sheet1']
    sheet.append(run_rst)
    wb.save(output_file_name)


def _load_config_files(file_list):
    loader = yaml.FullLoader
    file_config_dict = dict()
    if file_list:
        for file in file_list:
            with open(file, 'r', encoding='utf-8') as f:
                file_config_dict.update(yaml.load(f.read(), Loader=loader))
    return file_config_dict


output_file_name = 'denoising-ml.xlsx'
# mode = 'refine'
mode = 'model'
if __name__ == '__main__':
    model_name_list = ['HSD']
    sub_model_name_list = ['SASRec']  # ml-1m
    dataset = ['ml-10k']
    weight_decays = [0]

    for model in model_name_list:
        for data in dataset:
            for sub_model in sub_model_name_list:
                for w in weight_decays:
                    weight_decay = w
                    main(model, data, sub_model)
