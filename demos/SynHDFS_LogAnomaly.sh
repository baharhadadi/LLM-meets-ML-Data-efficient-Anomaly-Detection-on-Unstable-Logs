nvidia-smi
module load cuda

echo 'start running AD'

echo '*************0 percent test'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 0 --model_name=loganomaly --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 20 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --embedding_dim 300 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=128 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=3 --session_level=entry --output_dir=experimental_results/loganomaly/0/   --is_process --device cuda --semantics

echo '*************5 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 5 --model_name=loganomaly --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 20 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --embedding_dim 300 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=128 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=3 --session_level=entry --output_dir=experimental_results/loganomaly/5/   --is_process --device cuda --semantics

echo '*************10 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 10 --model_name=loganomaly --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 20 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --embedding_dim 300 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=128 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=3 --session_level=entry --output_dir=experimental_results/loganomaly/10/   --is_process --device cuda --semantics

echo '*************20 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 20 --model_name=loganomaly --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 20 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --embedding_dim 300 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=128 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=3 --session_level=entry --output_dir=experimental_results/loganomaly/20/   --is_process --device cuda --semantics

echo '*************30 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 30 --model_name=loganomaly --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 20 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --embedding_dim 300 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=128 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=3 --session_level=entry --output_dir=experimental_results/loganomaly/30/   --is_process --device cuda --semantics

echo '*************delete percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_type delete --model_name=loganomaly --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 20 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --embedding_dim 300 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=128 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=3 --session_level=entry --output_dir=experimental_results/loganomaly/remove/   --is_process --device cuda --semantics

echo '*************duplicate percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_type duplicate --model_name=loganomaly --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 20 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --embedding_dim 300 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=128 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=3 --session_level=entry --output_dir=experimental_results/loganomaly/duplicate/   --is_process --device cuda --semantics

echo '*************shuffle percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_type shuffle --model_name=loganomaly --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 20 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --embedding_dim 300 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=128 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=3 --session_level=entry --output_dir=experimental_results/loganomaly/shuffle/   --is_process --device cuda --semantics

echo 'finish'