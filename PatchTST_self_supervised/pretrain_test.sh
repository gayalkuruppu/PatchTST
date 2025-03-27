# python patchtst_pretrain.py --dset tuh_test --mask_ratio 0.4 --num_workers 8\
#      --batch_size 256 --context_points 25000 --patch_len 2500 --stride 2500 \
#      --n_epochs_pretrain 10

# python patchtst_pretrain.py --dset tuh_2000 --mask_ratio 0.4 --num_workers 16\
#      --batch_size 512 --context_points 25000 --patch_len 2500 --stride 2500 \
#      --n_epochs_pretrain 10 --use_multi_gpu # this gave the soft lock bug

# python patchtst_pretrain.py --dset tuh_test --mask_ratio 0.4 --num_workers 16\
#      --batch_size 256 --context_points 25000 --patch_len 2500 --stride 2500 \
#      --n_epochs_pretrain 10 --use_multi_gpu # single gpu 2mins for 1000 batches, 2:28 mins with multigpus for same num batches

# python patchtst_pretrain.py --dset tuh_test --mask_ratio 0.4 --num_workers 16\
#      --batch_size 512 --context_points 25000 --patch_len 2500 --stride 2500 \
#      --n_epochs_pretrain 10 --use_multi_gpu # single gpu 4:09 mins for 1000 batches,  mins with multigpus for same num batches

python patchtst_pretrain.py --dset tuh_test --mask_ratio 0.4 --num_workers 16\
     --batch_size 256 --context_points 25000 --patch_len 2500 --stride 2500 \
     --n_epochs_pretrain 10 --use_multi_gpu # single gpu mins for 1000 batches,  mins with multigpus for same num batches
