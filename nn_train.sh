rye run python bin/train.py k=3 num_layers=10 model=transformer fold=0 trainer.use_amp=false exp_name=transformer_0
rye run python bin/train.py k=3 num_layers=10 model=transformer fold=1 trainer.use_amp=false exp_name=transformer_1
rye run python bin/train.py k=3 num_layers=10 model=transformer fold=2 trainer.use_amp=false exp_name=transformer_2
rye run python bin/train.py k=3 num_layers=10 model=transformer fold=3 trainer.use_amp=false exp_name=transformer_3
rye run python bin/train.py k=3 num_layers=10 model=transformer fold=4 trainer.use_amp=false exp_name=transformer_4