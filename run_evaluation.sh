python get_response_openai_concurrent.py --env_file /example/dotenv/file/path/.env
python transform_format.py
# results/generation.jsonl + data/id2metric.json -> results/generation_trans.jsonl
CUDA_VISIBLE_DEVICES=0 python run_char_rm.py 
# results/generation_trans.jsonl + data/character_profiles.json -> results/evaluation
python compute_score.py