nvidia-smi
module load cuda

echo 'start running AD'

echo '*************0 percent test'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 0 --model_name=logrobust --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=128 --history_size=30 --lr=0.01 --seq_len=30 --session_level=entry --output_dir=experimental_results/logrobust/0/  --is_process --device cuda --semantics --input_size=300 --hidden_size=128 --num_layers=1 --dim_model=300 --adam_weight_decay=0.0001

echo '*************5 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 5 --model_name=logrobust --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=128 --history_size=30 --lr=0.01 --seq_len=30 --session_level=entry --output_dir=experimental_results/logrobust/5/  --is_process --device cuda --semantics --input_size=300 --hidden_size=128 --num_layers=1 --dim_model=300 --adam_weight_decay=0.0001

echo '*************10 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 10 --model_name=logrobust --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=128 --history_size=30 --lr=0.01 --seq_len=30 --session_level=entry --output_dir=experimental_results/logrobust/10/  --is_process --device cuda --semantics --input_size=300 --hidden_size=128 --num_layers=1 --dim_model=300 --adam_weight_decay=0.0001

echo '*************20 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 20 --model_name=logrobust --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=128 --history_size=30 --lr=0.01 --seq_len=30 --session_level=entry --output_dir=experimental_results/logrobust/20/  --is_process --device cuda --semantics --input_size=300 --hidden_size=128 --num_layers=1 --dim_model=300 --adam_weight_decay=0.0001

echo '*************30 percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_rate 30 --model_name=logrobust --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=128 --history_size=30 --lr=0.01 --seq_len=30 --session_level=entry --output_dir=experimental_results/logrobust/30/  --is_process --device cuda --semantics --input_size=300 --hidden_size=128 --num_layers=1 --dim_model=300 --adam_weight_decay=0.0001

echo '*************delete percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_type delete --model_name=logrobust --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=128 --history_size=30 --lr=0.01 --seq_len=30 --session_level=entry --output_dir=experimental_results/logrobust/remove/  --is_process --device cuda --semantics --input_size=300 --hidden_size=128 --num_layers=1 --dim_model=300 --adam_weight_decay=0.0001

echo '*************duplicate percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_type duplicate --model_name=logrobust --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=128 --history_size=30 --lr=0.01 --seq_len=30 --session_level=entry --output_dir=experimental_results/logrobust/duplicate/  --is_process --device cuda --semantics --input_size=300 --hidden_size=128 --num_layers=1 --dim_model=300 --adam_weight_decay=0.0001

echo '*************shuffle percent inject'
python baseline_main.py --folder=HDFS/ --log_file=HDFS.log --dataset_name=hdfs --inject --inject_type shuffle --model_name=logrobust --window_type=session --sample=sliding_window --window_size 30 --is_logkey --train_size=0.8 --train_ratio=1 --valid_ratio=0.1 --test_ratio=1 --max_epoch=20 --n_warm_up_epoch=0 --n_epochs_stop=10 --batch_size=128 --history_size=30 --lr=0.01 --seq_len=30 --session_level=entry --output_dir=experimental_results/logrobust/shuffle/  --is_process --device cuda --semantics --input_size=300 --hidden_size=128 --num_layers=1 --dim_model=300 --adam_weight_decay=0.0001

echo 'finish'