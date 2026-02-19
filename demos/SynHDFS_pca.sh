nvidia-smi
module load cuda

echo 'start running AD'

echo '*************0 percent test'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 0 --model_name=pca --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/pca/0 --is_process --device gpu

echo '*************5 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 5 --model_name=pca --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/pca/5 --is_process --device gpu

echo '*************10 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 10 --model_name=pca --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/pca/10 --is_process --device gpu

echo '*************20 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 20 --model_name=pca --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/pca/20 --is_process --device gpu

echo '*************30 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 30 --model_name=pca --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/pca/30 --is_process --device gpu

echo '*************delete percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_type delete --model_name=pca --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/pca/remove --is_process --device gpu

echo '*************duplicate percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_type duplicate --model_name=pca --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/pca/duplicate --is_process --device gpu

echo '*************shuffle percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject_type shuffle --model_name=pca --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=100 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=2024 --num_candidates=9 --history_size=30 --lr=0.001 --accumulation_step=1 --session_level=entry --output_dir=experimental_results/pca/shuffle --is_process --device gpu

echo 'finish'