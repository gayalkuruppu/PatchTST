# load data to test the dataloader

from src.data.pred_dataset import TUH_Dataset_Test
import numpy as np

def main():
    context_length = 25000
    patch_length = 2500

    tuh_data = TUH_Dataset_Test(
        root_path='/mnt/ssd_4tb_0/data/tuh_preprocessed_npy',
        data_path='',
        csv_path='../preprocessing/inputs/sub_list2.csv',
        features='M',
        scale=False,
        size=[context_length, 0, patch_length],
        use_time_features=False
    )

    print('len(tuh_data):', len(tuh_data))
    data_el = tuh_data[0]
    # print('data_el:', data_el)
    print(len(data_el))
    print('data_el[0]:', data_el[0][22500:, :])
    print('data_el[0].shape:', data_el[0].shape)
    print('data_el[1]:', data_el[1])
    print('data_el[1].shape:', data_el[1].shape)
    # data = np.memmap(data_el['data_path'], dtype='float32', mode='r', shape=(data_el['data_shape']))
    # print('data.shape:', data.shape)

if __name__=="__main__":
    main()