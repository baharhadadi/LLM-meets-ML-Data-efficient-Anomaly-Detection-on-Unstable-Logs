nvidia-smi
module load cuda

echo 'start running AD'

echo '*************0 percent test'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 0 --model_name=plelog --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/plelog/0/  --is_process --device cuda

echo '*************5 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 5 --model_name=plelog --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/plelog/5/  --is_process --device cuda

echo '*************10 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 10 --model_name=plelog --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/plelog/10/  --is_process --device cuda

echo '*************20 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 20 --model_name=plelog --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/plelog/20/  --is_process --device cuda

echo '*************30 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 30 --model_name=plelog --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/plelog/30/  --is_process --device cuda

echo '*************delete percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_type delete --model_name=plelog --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/plelog/remove/  --is_process --device cuda

echo '*************duplicate percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_type duplicate --model_name=plelog --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/plelog/duplicate/  --is_process --device cuda

echo '*************shuffle percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_type shuffle --model_name=plelog --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/plelog/shuffle/  --is_process --device cuda

echo 'finish'