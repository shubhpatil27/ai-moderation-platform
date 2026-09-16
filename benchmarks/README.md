# Performance Benchmarks

All benchmark results in this directory are generated from
reproducible scripts or load-test commands.

## Model Runtime Benchmark

Run:

python -m scripts.benchmark

Configuration:

- Warm-up runs: 10
- Measured runs: 100
- Device: CPU
- Model: unitary/toxic-bert

Results:

benchmarks/raw/model_runtime.csv


## API Load Benchmark

Run:

python -m locust -f load_tests/locustfile.py --headless -u 25 -r 5 -t 2m --host http://127.0.0.1:8000 --csv benchmarks/raw/onnx_25_users

Configuration:

- Concurrent users: 25
- Spawn rate: 5 users/sec
- Duration: 2 minutes
- Endpoint: POST /moderate
- Deployment: Docker Compose
- Database: PostgreSQL

Results:

benchmarks/raw/onnx_25_users_stats.csv