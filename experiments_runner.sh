c_batch=(hardcoded)
c_epochs=(hardcoded)
c_random_seed=(hardcoded)

c_lr=(0.0005 0.001 0.002)
c_patch_size=(256)
c_patch_step_train=(128)
c_val_set_ratio=(0.1)
c_test_set_ratio=(0.15)

project_name="DRIVE_ONLY"
limits=0

for batch in "${c_batch[@]}"; do
    for epochs in "${c_epochs[@]}"; do
        for lr in "${c_lr[@]}"; do
            for patch_size in "${c_patch_size[@]}"; do
                for patch_step_train in "${c_patch_step_train[@]}"; do
                    for random_seed in "${c_random_seed[@]}"; do
                        for val_set_ratio in "${c_val_set_ratio[@]}"; do
                            for test_set_ratio in "${c_test_set_ratio[@]}"; do
                                echo batch_size:"$batch" epochs:"$epochs" patch_size:"$patch_size" patch_step_train:"$patch_step_train" \
                                    lr:"$lr" random_seed:"$random_seed" val_set_ratio:"$val_set_ratio" test_set_ratio:"$test_set_ratio"
                                
                                python main.py --learning_rate=$lr --patch_size=$patch_size --patch_step_train=$patch_step_train \
                                            --val_set_ratio=$val_set_ratio --test_set_ratio=$test_set_ratio --project_name=$project_name \
                                            --limits=$limits
                            done
                        done
                    done
                done
            done
        done
    done
done
