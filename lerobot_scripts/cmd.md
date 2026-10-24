




## HF commands

lerobot-train \
  --dataset.repo_id="pick_and_place_red_lego" \
  --dataset.root="data/pick_and_place_red_lego" \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test_2_dummyy \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --steps=300000 \
  --policy.push_to_hub=false


  lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=pick_and_place_red_lego   --dataset.root=data/pick_and_place_red_lego_v2   --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'   --batch_size=8   --steps=20000   --output_dir=outputs/train/my_smolvla   --job_name=my_smolvla_training   --policy.device=cuda   --policy.push_to_hub=false   --wandb.enable=true

##### Training
### Act
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id="pick_and_place_red_lego" \
  --dataset.root="data/pick_and_place_red_lego" \
  --output_dir=outputs/train/act_so101_test_dummy \
  --job_name=act_so101_test \
  --wandb.enable=false \
  --steps=300000 


### SmolVLA
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="pick_and_place_red_lego" \
  --dataset.root="data/pick_and_place_red_lego" \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla_2_cam_data_v1 \
  --job_name=my_smolvla_training_2_cam_data_v1 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --wandb.enable=true \
  --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist.left":"observation.images.camera2"}'

### Pi-0
lerobot-train   
--policy.type=pi0   
--dataset.repo_id="pick_and_place_red_lego"   
--dataset.root="data/pick_and_place_red_lego_v2"   
--job_name=pi0_training   
--output_dir=outputs/pi0_training   
--policy.pretrained_path=lerobot/pi0_base  
--policy.compile_model=true   
--policy.gradient_checkpointing=true   
--policy.dtype=bfloat16   
--steps=20000   
--policy.device=cuda   
--policy.push_to_hub=false  
--batch_size=32   
--wandb.enable=true 



lerobot-train \
--dataset.repo_id="pick_and_place_red_lego"   \
--dataset.root="data/pick_and_place_red_lego_v2"  \
--policy.type=pi0 \
--output_dir=outputs/train/pi0_training  \
--job_name=pi0_training \
--policy.pretrained_path=lerobot/pi0_base \
--policy.repo_id=your_repo_id \
--policy.compile_model=true \
--policy.gradient_checkpointing=true \
--policy.dtype=bfloat16 \
--policy.freeze_vision_encoder=false \
--policy.train_expert_only=false \
--policy.push_to_hub=false \
--steps=3000 \
--policy.device=cuda \
--batch_size=32



lerobot-train \
  --policy.type=pi0 \
  --dataset.repo_id="pick_and_place_red_lego" \
  --dataset.root="" \
  --job_name=pi0_training \
  --output_dir=outputs/pi0_training \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --steps=20000 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=32 \
  --wandb.enable=true 




###### Testing
### test act 
lerobot-record  \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ wrist.left: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30 },}" \
  --robot.id=my_follower_arm \
  --dataset.repo_id="eval/eval_pick_and_place_red_lego"\
  --dataset.root="data/eval_act_fg_jun10_2" \
  --dataset.push_to_hub="false"\
  --dataset.single_task="Pick the red lego block and put it in the box"  \
  --dataset.num_episodes=10 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --policy.path="outputs/train/act_so101_test/checkpoints/last/pretrained_model"\
  --policy.push_to_hub="false"\
  --policy.temporal_ensemble_coeff=0.1\
  --policy.n_action_steps=1

### test act without temporal
lerobot-record  \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras="{ wrist.left: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30 },}" \
  --robot.id=my_follower_arm \
  --dataset.repo_id="eval/eval_pick_and_place_red_lego"\
  --dataset.root="data/eval_pick_and_place_red_lego_act_v1_test_2" \
  --dataset.push_to_hub="false"\
  --dataset.single_task="Pick the red lego block and put it in the box"  \
  --dataset.num_episodes=1 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --policy.path="outputs/train/act_so101_test/checkpoints/last/pretrained_model"\
  --policy.push_to_hub="false"\





### test act v2
lerobot-record  \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30 },}" \
  --robot.id=my_follower_arm \
  --dataset.repo_id="eval/eval_pick_and_place_red_lego"\
  --dataset.root="data/eval_pick_and_place_red_lego_act_v2" \
  --dataset.push_to_hub="false"\
  --dataset.single_task="Pick the red lego block and put it in the box"  \
  --dataset.num_episodes=1 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --policy.path="outputs/train/act_so101_test_2/checkpoints/last/pretrained_model"



#### test smolvla

lerobot-record  \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras="{ camera2: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30 },}" \
  --robot.id=my_follower_arm \
  --dataset.repo_id="eval/eval_pick_and_place_red_lego"\
  --dataset.root="data/eval_2_pick_and_place_red_lego_smolvla_dummy" \
  --dataset.push_to_hub="false"\
  --dataset.single_task="Pick the red lego block and put it in the box"  \
  --dataset.num_episodes=1 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --policy.path="outputs/train/my_smolvla_2_cam/checkpoints/last/pretrained_model"


lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="pick_and_place_red_lego" \
  --dataset.root="data/pick_and_place_red_lego_v2" \
  --batch_size=8 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla_2_cam \
  --job_name=my_smolvla_training_2_cam \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --wandb.enable=true \
  --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'



lerobot-rollout \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras="{ wrist.left: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30 },}" \
  --strategy.type=base \
  --policy.path="outputs/train/act_so101_test/checkpoints/last/pretrained_model" \
  --task="Pick the red lego block and put it in the box" \
  --duration=30

