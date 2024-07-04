for ((i=1;i<=10;i++)); do
clear
let seed=$i*100
mkdir -p ./deep_dialog/checkpoints/movie_esdp_DQN/run_$i
python run.py --agt 9 \
--usr 1 \
--max_turn 40 \
--movie_kb_path ./deep_dialog/data/movie_kb.1k.p \
--dqn_hidden_size 80 \
--experience_replay_pool_size 5000 \
--episodes 300 \
--simulation_epoch_size 100 \
--run_mode 3 \
--act_level 0 \
--slot_err_prob 0.05 \
--intent_err_prob 0.00 \
--batch_size 16 \
--goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
--warm_start 1 \
--warm_start_epochs 120 \
--planning_steps 0 \
--write_model_dir ./deep_dialog/checkpoints/movie_esdp_DQN/run_$i \
--torch_seed $seed \
--grounded 0 \
--boosted 0 \
--train_world_model 0 \
--user_abort 0 \
--alpha 0.7 \
--beta 20.0 \
--topk 3
done

python fetch_top.py --source_path ./deep_dialog/checkpoints/movie_esdp_07_20


