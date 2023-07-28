pre_train_model_dict = {
    'ml-100k': {
        'GRU4Rec': 'saved/GRU4Rec-Jul-16-2023_13-52-57.pth',
        'NARM': 'saved/NARM-Jul-16-2023_13-54-53.pth',
        'Caser': 'saved/Caser-Jul-16-2023_13-57-33.pth',
        'BERT4Rec': 'saved/BERT4Rec-Jul-16-2023_14-00-43.pth',
        'SASRec': 'saved/SASRec-Jul-16-2023_14-06-45.pth',
        'STAMP': 'saved/STAMP-Jul-16-2023_16-42-36.pth'
        # 将GRU4Rec训练好的embedding又拿去训练GRU4Rec，效果不变
    },

    'amazon-beauty': {
        'GRU4Rec': 'saved/GRU4Rec-Jul-16-2023_13-53-19.pth',
        'NARM': 'saved/NARM-Jul-16-2023_13-55-17.pth',
        'Caser': 'saved/Caser-Jul-16-2023_13-57-57.pth',
        'BERT4Rec': 'saved/BERT4Rec-Jul-16-2023_14-01-11.pth',
        'SASRec': 'saved/SASRec-Jul-16-2023_14-07-48.pth',
        'STAMP': 'saved/STAMP-Jul-16-2023_16-44-44.pth'
    },

    'amazon-sports-outdoors': {
        'GRU4Rec': 'saved/GRU4Rec-Jul-16-2023_13-54-21.pth',
        'NARM': 'saved/NARM-Jul-16-2023_13-56-51.pth',
        'Caser': 'saved/Caser-Jul-16-2023_13-59-33.pth',
        'BERT4Rec': 'saved/BERT4Rec-Jul-16-2023_14-04-08.pth',
        'SASRec': 'saved/SASRec-Jul-16-2023_14-50-14.pth',
        'STAMP': 'saved/STAMP-Jul-16-2023_17-06-35.pth',
        # 'DSAN': 'saved/DSAN-May-19-2022_23-25-56.pth'
    },
    'yelp': {
        'GRU4Rec': 'saved/GRU4Rec-Jul-16-2023_13-54-20.pth',
        'NARM': 'saved/NARM-Jul-16-2023_13-57-10.pth',
        'Caser': 'saved/Caser-Jul-16-2023_14-00-14.pth',
        'BERT4Rec': 'saved/BERT4Rec-Jul-16-2023_14-02-43.pth',
        'SASRec': 'saved/SASRec-Jul-16-2023_14-16-37.pth',
        'STAMP': 'saved/STAMP-Jul-16-2023_15-15-34.pth'
    },

    'ml-1m': {
        'GRU4Rec': 'saved/GRU4Rec-Jul-16-2023_13-55-54.pth',
        'NARM': 'saved/NARM-Jul-16-2023_13-59-07.pth',
        'Caser': 'saved/Caser-Jul-16-2023_14-01-30.pth',
        'BERT4Rec': 'saved/BERT4Rec-Jul-16-2023_14-08-23.pth',
        'SASRec': 'saved/SASRec-Jul-16-2023_15-03-26.pth',
        'STAMP': 'saved/STAMP-Jul-16-2023_15-17-09.pth'
        # 'STAMP': 'saved/BERT4Rec-May-17-2022_13-16-38.pth'  # 用bert4rec的embedding比用stamp自己学出来的性能要好
    }

}

pre_train_model_dict0715 = {
    'ml-100k': {
        # 'GRU4Rec': 'saved/GRU4Rec-Jul-09-2023_12-45-50.pth', # old
        # 'GRU4Rec': 'saved/GRU4Rec-Jul-15-2023_02-11-05.pth',  # new ce
        'GRU4Rec': 'saved/GRU4Rec-Jul-15-2023_11-11-16.pth',  # new bpr
        'Caser': 'saved/Caser-Jul-15-2023_11-41-26.pth',
        'SASRec': 'saved/SASRec-Jul-15-2023_11-12-44.pth',
        'BERT4Rec': 'saved/BERT4Rec-Jul-15-2023_11-32-12.pth',
        # 将GRU4Rec训练好的embedding又拿去训练GRU4Rec，效果不变
    },
    'amazon-sports-outdoors': {
        'GRU4Rec':'saved/GRU4Rec-Jul-15-2023_11-15-20.pth',
        'Caser':'saved/Caser-Jul-15-2023_12-13-34.pth',
        'SASRec':'saved/SASRec-Jul-15-2023_11-18-32.pth',
        'BERT4Rec':'saved/BERT4Rec-Jul-15-2023_12-05-13.pth'
    },
    'yelp': {
        'GRU4Rec':'saved/GRU4Rec-Jul-15-2023_11-16-52.pth',
        'Caser':'saved/Caser-Jul-15-2023_12-15-38.pth',
        'SASRec':'saved/SASRec-Jul-15-2023_11-38-09.pth',
        'BERT4Rec':'saved/BERT4Rec-Jul-15-2023_12-08-00.pth'
    },
    'amazon-beauty': {
        'GRU4Rec':'saved/GRU4Rec-Jul-15-2023_11-12-17.pth',
        'Caser':'saved/Caser-Jul-15-2023_11-42-34.pth',
        'SASRec':'saved/SASRec-Jul-15-2023_11-20-32.pth',
        'BERT4Rec':'saved/BERT4Rec-Jul-15-2023_11-39-23.pth'
    },
    'amazon-clothing-shoes-jewelry': {
        'GRU4Rec': 'saved/GRU4Rec-May-17-2022_13-32-16.pth',
        'NARM': 'saved/NARM-May-17-2022_13-34-29.pth',
        'Caser': 'saved/Caser-May-17-2022_13-37-12.pth',
        'BERT4Rec': 'saved/BERT4Rec-May-17-2022_13-40-44.pth',
        # 'BERT4Rec': 'saved/Caser-May-17-2022_13-37-12.pth',
        'SASRec': 'saved/SASRec-May-17-2022_13-46-02.pth',
        'STAMP': 'saved/STAMP-May-17-2022_15-03-23.pth'
        #使用 Caser的embedding尝试训练Bert4Rec
    },
    'ml-1m': {
        'GRU4Rec':'saved/GRU4Rec-Jul-15-2023_11-11-44.pth',
        'Caser':'saved/Caser-Jul-15-2023_11-41-53.pth',
        'SASRec':'saved/SASRec-Jul-15-2023_11-13-29.pth',
        'BERT4Rec':'saved/BERT4Rec-Jul-15-2023_11-32-44.pth'
        # 'STAMP': 'saved/BERT4Rec-May-17-2022_13-16-38.pth'  # 用bert4rec的embedding比用stamp自己学出来的性能要好
    }
}
pre_train_model_dict0 = {
    'amazon-sports-outdoors': {
        'GRU4Rec': 'saved/GRU4Rec-May-17-2022_13-14-44.pth',
        'NARM': 'saved/NARM-May-17-2022_13-16-07.pth',
        'Caser': 'saved/Caser-May-17-2022_13-17-42.pth',
        'BERT4Rec': 'saved/BERT4Rec-May-17-2022_13-20-26.pth',
        'SASRec': 'saved/SASRec-Jul-16-2023_02-01-29.pth',
        'STAMP': 'saved/STAMP-May-17-2022_14-20-57.pth',
        'DSAN': 'saved/DSAN-May-19-2022_23-25-56.pth'
    },
    'ml-100k': {
        'GRU4Rec': 'saved/GRU4Rec-May-17-2022_13-13-42.pth',
        'NARM': 'saved/NARM-May-19-2022_19-31-33.pth',
        'Caser': 'saved/Caser-May-17-2022_13-13-56.pth',
        'BERT4Rec': 'saved/BERT4Rec-May-17-2022_13-14-04.pth',
        'SASRec': 'saved/SASRec-May-17-2022_13-14-14.pth',
        'STAMP': 'saved/STAMP-May-17-2022_13-14-49.pth'
        # 将GRU4Rec训练好的embedding又拿去训练GRU4Rec，效果不变
    },
    'yelp': {
        'GRU4Rec': 'saved/GRU4Rec-May-17-2022_13-15-01.pth',
        'NARM': 'saved/NARM-May-17-2022_13-17-59.pth',
        'Caser': 'saved/Caser-May-17-2022_13-26-00.pth',
        'BERT4Rec': 'saved/BERT4Rec-May-17-2022_13-30-05.pth',
        'SASRec': 'saved/SASRec-May-17-2022_13-45-49.pth',
        'STAMP': 'saved/STAMP-May-17-2022_14-27-11.pth'
    },
    'amazon-beauty': {
        'GRU4Rec': 'saved/GRU4Rec-May-17-2022_13-13-26.pth',
        'NARM': 'saved/NARM-May-17-2022_13-14-20.pth',
        'Caser': 'saved/Caser-May-17-2022_13-16-13.pth',
        'BERT4Rec': 'saved/BERT4Rec-May-17-2022_13-17-49.pth',
        'SASRec': 'saved/SASRec-May-17-2022_13-21-48.pth',
        'STAMP': 'saved/STAMP-May-17-2022_14-05-50.pth'
    },
    'amazon-clothing-shoes-jewelry': {
        'GRU4Rec': 'saved/GRU4Rec-May-17-2022_13-32-16.pth',
        'NARM': 'saved/NARM-May-17-2022_13-34-29.pth',
        'Caser': 'saved/Caser-May-17-2022_13-37-12.pth',
        'BERT4Rec': 'saved/BERT4Rec-May-17-2022_13-40-44.pth',
        # 'BERT4Rec': 'saved/Caser-May-17-2022_13-37-12.pth',
        'SASRec': 'saved/SASRec-May-17-2022_13-46-02.pth',
        'STAMP': 'saved/STAMP-May-17-2022_15-03-23.pth'
        #使用 Caser的embedding尝试训练Bert4Rec
    },
    'ml-1m': {
        'GRU4Rec': 'saved/GRU4Rec-May-17-2022_13-13-41.pth',
        'NARM': 'saved/NARM-May-17-2022_13-14-36.pth',
        'Caser': 'saved/Caser-May-17-2022_13-15-33.pth',
        'BERT4Rec': 'saved/BERT4Rec-May-17-2022_13-16-38.pth',
        'SASRec': 'saved/SASRec-May-17-2022_13-27-12.pth',
        'STAMP': 'saved/STAMP-May-17-2022_13-39-43.pth'
        # 'STAMP': 'saved/BERT4Rec-May-17-2022_13-16-38.pth'  # 用bert4rec的embedding比用stamp自己学出来的性能要好
    }

}