for ((i=1;i<=30;i++));
do
  clear
  let seed=$i*100
  mkdir -p ./deep_dialog/checkpoints/taxi_esdp_07_20/run_$i
  python run.py \
  --agt 13 \
  --usr 2 \
  --max_turn 40 \
  --movie_kb_path ./deep_dialog/data_taxi/taxi.kb.2k.v1.p \
  --goal_file_path ./deep_dialog/data_taxi/user_goals_first.part.taxi.v4.p \
  --slot_set ./deep_dialog/data_taxi/taxi_slots.txt \
	--act_set ./deep_dialog/data_taxi/dia_acts.txt \
	--dict_path ./deep_dialog/data_taxi/slot_dict.v1.p \
  --dqn_hidden_size 80 \
  --experience_replay_pool_size 5000 \
  --episodes 300 \
  --simulation_epoch_size 100 \
  --run_mode 3 \
  --act_level 0 \
  --slot_err_prob 0.05 \
  --intent_err_prob 0.00 \
  --batch_size 16 \
  --warm_start 1 \
  --warm_start_epochs 120 \
  --planning_steps 0 \
  --write_model_dir ./deep_dialog/checkpoints/taxi_esdp_07_20/run_$i \
  --torch_seed $seed \
  --grounded 0 \
  --boosted 0 \
  --train_world_model 0 \
  --user_abort 0 \
  --alpha 0.7 \
  --beta 20.0 \
  --topk 3
done

python fetch_top.py --source_path ./deep_dialog/checkpoints/taxi_esdp_07_20


