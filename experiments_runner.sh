# WANDB project name with experiments
project_name="Patch_size_experiments"

# Draw dataset each time this parameter is changed
c_patch_size=(256 128 64)
c_patch_step_train=(256) 
c_test_set_ratio=(0.15)

# Run experiments on already prepared dataset
c_model_name=(base_unet r2_unet attention_unet ladder_net sa_unet)
c_val_set_ratio=(0.15)
c_lr=(0.001)


for patch_size in "${c_patch_size[@]}"; do
    for patch_step_train in "${c_patch_step_train[@]}"; do
        for test_set_ratio in "${c_test_set_ratio[@]}"; do
            echo Preparing new dataset
            python main.py --prepare_new_dataset=0
            
            for model_name in "${c_model_name[@]}"; do            
                for val_set_ratio in "${c_val_set_ratio[@]}"; do
                    for lr in "${c_lr[@]}"; do
                            echo patch_size:"$patch_size" patch_step_train:"$patch_step_train" model_name:"$model_name"\
                                test_set_ratio:"$test_set_ratio" lr:"$lr" val_set_ratio:"$val_set_ratio" 
                            
                            python main.py --learning_rate=$lr --patch_size=$patch_size --patch_step_train=$patch_size --model_name=$model_name \
                                        --val_set_ratio=$val_set_ratio --test_set_ratio=$test_set_ratio --project_name=$project_name
                    done
                done
            done
        done
    done
done
