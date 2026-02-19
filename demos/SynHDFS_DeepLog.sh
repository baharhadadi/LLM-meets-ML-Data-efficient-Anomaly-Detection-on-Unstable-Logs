nvidia-smi
module load cuda

echo 'start running AD'

echo '*************0 percent test'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --input_size=300 --dataset_name=hdfs --inject --inject_rate 0 --model_name=deeplog --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=10 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=256 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=2 --session_level=entry --output_dir=experimental_results/deeplog/0/      --is_process --device cuda

echo '*************5 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --input_size=300 --dataset_name=hdfs --inject --inject_rate 5 --model_name=deeplog --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=10 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=256 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=2 --session_level=entry --output_dir=experimental_results/deeplog/5/      --is_process --device cuda

echo '*************10 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --input_size=300 --dataset_name=hdfs --inject --inject_rate 10 --model_name=deeplog --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=10 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=256 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=2 --session_level=entry --output_dir=experimental_results/deeplog/10/      --is_process --device cuda

echo '*************20 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --input_size=300 --dataset_name=hdfs --inject --inject_rate 20 --model_name=deeplog --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=10 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=256 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=2 --session_level=entry --output_dir=experimental_results/deeplog/20/      --is_process --device cuda

echo '*************30 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --input_size=300 --dataset_name=hdfs --inject --inject_rate 30 --model_name=deeplog --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=10 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=256 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=2 --session_level=entry --output_dir=experimental_results/deeplog/30/      --is_process --device cuda

echo '*************delete percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --input_size=300 --dataset_name=hdfs --inject --inject_type delete --model_name=deeplog --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=10 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=256 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=2 --session_level=entry --output_dir=experimental_results/deeplog/remove/      --is_process --device cuda

echo '*************duplicate percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --input_size=300 --dataset_name=hdfs --inject --inject_type duplicate --model_name=deeplog --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=10 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=256 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=2 --session_level=entry --output_dir=experimental_results/deeplog/duplicate/      --is_process --device cuda

echo '*************shuffle percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --input_size=300 --dataset_name=hdfs --inject --inject_type shuffle --model_name=deeplog --window_type=session --sample=sliding_window --embedding_dim 300 --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=10 --n_warm_up_epoch=0 --n_epochs_stop=5 --batch_size=256 --num_candidates=9 --history_size=15 --lr=0.0001 --accumulation_step=2 --session_level=entry --output_dir=experimental_results/deeplog/shuffle/      --is_process --device cuda

echo 'finish'