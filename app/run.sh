python -m app.main \
  --fname configs/train/metaworld_predictor.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3  \
  --wandb-run-name my_run_name