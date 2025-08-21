
# command
```
python experiments/robot/libero/run_libero_dataset.py \  
--model_family openvla \  
--pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \  
--task_suite_name libero_object \  
--center_crop True \  
--num_trials_per_task 5 \  
--seed 42 \  
--log_dataset True \  
--dataset_dir ./rollouts_libero  
```