import os


from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    set_seed,
    Trainer,
    TrainingArguments,
    PatchTSTForPretraining
)
import numpy as np
import pandas as pd
# from dataset import IMUDataset
from src.data.pred_dataset import TUH_Dataset_Test
import torch
import json


# from Ego4D.ego4d_dataset import Ego4dDatasetUnsupervised
# from transformers import AutoImageProcessor

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

import os
from getpass import getpass
os.environ["NEPTUNE_API_TOKEN"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjNzI3OWFjMy02ZWQ5LTQzMTctOGYxMC1iMDliZTk3N2Y1YzMifQ=="
os.environ["NEPTUNE_PROJECT"] = "gayalkuruppu/tuh-ssl-test"

def main():

    np.random.seed(42)

    # image_processor = AutoImageProcessor.from_pretrained(
    #     "MCG-NJU/videomae-base-ssv2")
    
    # ego4d_data = Ego4dDatasetUnsupervised(image_processor,5,10,max_n_windows_per_video=None)



    context_length = 2500#512
    forecast_horizon = 96
    patch_length = 250#16
    patch_stride = 250
    num_workers = 16 
    batch_size = 128#512 
    avg_imu_length = 512


    tuh_data = TUH_Dataset_Test(
        root_path='/mnt/ssd_4tb_0/data/tuh_preprocessed_npy_test',#'/mnt/ssd_4tb_0/data/tuh_preprocessed_npy',
        data_path='',
        csv_path='../preprocessing/inputs/sub_list2.csv',
        features='M',
        scale=False,
        size=[context_length, 0, patch_length],
        use_time_features=False
    )

    tuh_eval_data = TUH_Dataset_Test(
        root_path='/mnt/ssd_4tb_0/data/tuh_preprocessed_npy_test',#'/mnt/ssd_4tb_0/data/tuh_preprocessed_npy',
        data_path='',
        csv_path='../preprocessing/inputs/sub_list2.csv',
        features='M',
        scale=False,
        size=[context_length, 0, patch_length],
        use_time_features=False,
        split='val'
    )

    # classes =  np.unique([path.split('/')[6] for path in ssl_data])


    # train_dataset = IMUDataset(
    #     train_data,
    #     classes= classes,
    #     avg_imu_length=avg_imu_length
    # )
    # val_dataset = IMUDataset(
    #     validation_data,
    #     classes= classes,
    #     avg_imu_length=avg_imu_length
    # )
    
    config = PatchTSTConfig(
        num_input_channels=19, #6,
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
        mask_type='random',
        random_mask_ratio=0.4,
        use_cls_token=True,
        # prediction_length=forecast_horizon,
    )
    model = PatchTSTForPretraining(config)

    training_args = TrainingArguments(
        output_dir='/home/gayal/ssl-project/PatchTST/PatchTST_self_supervised/saved_models/test_run_10_recordings_10k/outputs', #"./checkpoint/patchtst-ego4d/pretrain/output/",
        overwrite_output_dir=True,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        # lr_scheduler_kwargs={'eta_min':1e-7, },
        # num_train_epochs=100,#100,
        max_steps=10000,
        do_eval=True,
        eval_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        save_strategy="steps",
        logging_strategy="steps",
        save_steps=0.05,
        logging_steps=1,
        eval_steps=100,
        # save_total_limit=3,
        logging_dir='/home/gayal/ssl-project/PatchTST/PatchTST_self_supervised/saved_models/test_run_10_recordings_10k/logs',#"./checkpoint/patchtst-ego4d/pretrain/logs/",  
        # load_best_model_at_end=True,  
        # metric_for_best_model="loss",  
        greater_is_better=False,  # For loss
        # label_names=["future_values"],
        report_to="neptune"
    )
    

    # # Create the early stopping callback
    # early_stopping_callback = EarlyStoppingCallback(
    #     early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
    #     early_stopping_threshold=0.0001,  # Minimum improvement required to consider as improvement
    # )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tuh_data,#ego4d_data,
        # data_collator=CustomDataCollator()
        eval_dataset=tuh_eval_data,
        # callbacks=[early_stopping_callback],
        # compute_metrics=compute_metrics,
    )

    
    trainer.train()

if __name__ == "__main__":
    main()

# NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python IMU_SSL.py