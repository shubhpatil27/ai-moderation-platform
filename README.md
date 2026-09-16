# 🛡️ AI Moderation Platform

### Production-Oriented, Personalized AI Content Moderation with MLOps

An end-to-end machine learning platform for **real-time toxic content moderation**, designed around a practical social-media use case:

> Give users control over what kinds of harmful comments they want automatically filtered from posts, stories, streams, and other interactive content.

This project goes beyond running a pretrained NLP model behind an API.

It implements a complete ML systems workflow involving:

- 🤖 Transformer-based toxicity classification
- ⚡ ONNX Runtime inference optimization
- 📦 INT8 model quantization
- 🎛️ User-specific moderation preferences
- 🌐 FastAPI model serving
- 🐘 PostgreSQL persistence
- 🐳 Dockerized deployment
- 🧪 Automated testing
- 🚦 Automated model promotion gates
- 📊 MLflow experiment tracking
- 🗂️ MLflow Model Registry
- 👥 Human feedback collection
- 🔥 Locust load testing
- 📈 Reproducible benchmarking
- 🔄 CI with GitHub Actions

---

# 🎯 Problem

Social platforms process enormous volumes of comments in real time.

Traditional moderation systems often make a single platform-wide decision:

```text
Comment
   ↓
Moderation System
   ↓
ALLOW / HIDE
```

But different users may have different preferences.

One creator may want insults automatically hidden while another may only want threats or severe toxicity filtered.

This project explores a more personalized architecture:

```text
                         User Preferences
                               ↓
Incoming Comment → Toxicity Model → Policy Engine
                                      ↓
                              ALLOW / REVIEW / HIDE
```

The ML model estimates toxicity categories.

The **policy layer decides what those predictions mean for a particular user**.

This separation allows moderation behavior to change without retraining the underlying model every time a user changes a preference.

---

# 🧠 ML Model

The platform uses:

**`unitary/toxic-bert`**

for multi-label toxicity classification.

The model produces probabilities for six categories:

| Category | Meaning |
|---|---|
| `toxic` | General toxic language |
| `severe_toxic` | Extremely toxic language |
| `obscene` | Obscene content |
| `threat` | Threatening language |
| `insult` | Insulting language |
| `identity_hate` | Identity-targeted hateful language |

Example model output:

```json
{
  "toxic": 0.9878,
  "severe_toxic": 0.0425,
  "obscene": 0.7727,
  "threat": 0.0013,
  "insult": 0.9573,
  "identity_hate": 0.0133
}
```

The model output is deliberately separated from the final moderation policy.

---

# 🎛️ Personalized Moderation

Users have individual moderation preferences stored in PostgreSQL.

Example:

```json
{
  "toxic": true,
  "severe_toxic": true,
  "obscene": false,
  "threat": true,
  "insult": true,
  "identity_hate": true
}
```

This means two users can receive the **same model prediction** while getting different moderation outcomes.

Conceptually:

```text
                  Toxicity Model
                       ↓
              Category Probabilities
                       ↓
              User Preferences
                       ↓
                 Policy Engine
                       ↓
             ALLOW / REVIEW / HIDE
```

This makes the system suitable as the backend for a future social-media setting such as:

> “Automatically filter harmful comments according to my preferences.”

---

# 🏗️ System Architecture

```text
                           ┌───────────────────┐
                           │      Client       │
                           │ Social / Web App  │
                           └─────────┬─────────┘
                                     │
                                     ▼
                           ┌───────────────────┐
                           │      FastAPI      │
                           │   REST Service    │
                           └─────────┬─────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
          ┌──────────────────┐              ┌──────────────────┐
          │    PostgreSQL    │              │  ML Inference    │
          │                  │              │                  │
          │ Users            │              │ Toxic-BERT       │
          │ Preferences      │              │ ONNX Runtime     │
          │ Predictions      │              │ CPU Inference    │
          │ Feedback         │              └────────┬─────────┘
          └──────────────────┘                       │
                                                    ▼
                                          ┌──────────────────┐
                                          │  Policy Engine   │
                                          │                  │
                                          │ User Preferences │
                                          │ + ML Scores      │
                                          └────────┬─────────┘
                                                   │
                                                   ▼
                                          ALLOW / REVIEW / HIDE
```

---

# 🔄 Request Flow

A moderation request travels through the system as follows:

```text
POST /moderate
      ↓
Validate request
      ↓
Load user from PostgreSQL
      ↓
Load user's moderation preferences
      ↓
Tokenize comment
      ↓
Run Toxic-BERT inference
      ↓
Generate six toxicity probabilities
      ↓
Apply personalized moderation policy
      ↓
Store prediction in PostgreSQL
      ↓
Return decision
```

This is an end-to-end ML-backed application rather than an isolated notebook experiment.

---

# ⚡ Model Optimization

One of the central engineering goals was reducing inference latency.

The project therefore evaluates three inference configurations:

```text
PyTorch FP32
     ↓
ONNX FP32
     ↓
ONNX INT8
```

Rather than assuming optimization helped, each stage was benchmarked.

---

# 🚀 PyTorch → ONNX Runtime

The original Transformer model was exported from PyTorch to ONNX.

Prediction sanity checks were performed between the two runtimes before benchmarking.

For the tested comments, the returned probabilities matched at the four-decimal precision exposed by the predictor.

Example:

```text
CATEGORY             PYTORCH      ONNX

toxic                 0.9878      0.9878
severe_toxic          0.0425      0.0425
obscene               0.7727      0.7727
threat                0.0013      0.0013
insult                 0.9573      0.9573
identity_hate          0.0133      0.0133
```

This provided a basic correctness check before treating ONNX as an optimization candidate.

---

# 📊 Model Runtime Benchmark

A reproducible local CPU benchmark compares the inference runtimes.

One measured run produced:

| Runtime | Average | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 38.34 ms | 36.61 ms | 43.62 ms | 73.24 ms |
| ONNX FP32 | 15.88 ms | 14.31 ms | 23.29 ms | 25.71 ms |
| ONNX INT8 | **6.84 ms** | **5.86 ms** | **10.71 ms** | **14.15 ms** |

Observed speedups in that run:

```text
ONNX FP32 vs PyTorch  → 2.41×
ONNX INT8 vs PyTorch → 5.60×
ONNX INT8 vs FP32    → 2.32×
```

These measurements are environment-specific local CPU benchmark results and are not intended as universal performance claims.

Benchmark results are automatically written to:

```text
benchmarks/raw/model_runtime.csv
```

---

# 📦 INT8 Quantization

The ONNX model was dynamically quantized from FP32 to INT8.

Measured artifact sizes:

```text
ONNX FP32
417.86 MB
     ↓
INT8 Quantization
     ↓
ONNX INT8
105.14 MB
```

### Model size reduction

**74.84%**

| Model | Size |
|---|---:|
| ONNX FP32 | 417.86 MB |
| ONNX INT8 | **105.14 MB** |

But a smaller and faster model is not automatically a better production model.

Quantization can change predictions.

Therefore the INT8 model had to pass a proper quality evaluation before promotion.

---

# 🧪 Model Quality Evaluation

A deterministic **5,000-comment evaluation subset** was constructed from the Jigsaw Toxic Comment dataset.

The subset intentionally preserves positive examples for rare toxicity categories.

Evaluation coverage:

| Category | Positive Examples |
|---|---:|
| Toxic | 2,726 |
| Severe Toxic | 801 |
| Obscene | 2,080 |
| Threat | 478 |
| Insult | 2,028 |
| Identity Hate | 726 |

Both ONNX FP32 and ONNX INT8 were evaluated on the **same comments**.

---

# 📈 FP32 vs INT8 Quality

Measured F1 scores:

| Category | FP32 F1 | INT8 F1 | Change |
|---|---:|---:|---:|
| Toxic | 0.9696 | 0.9680 | -0.0016 |
| Severe Toxic | 0.5443 | 0.5402 | -0.0042 |
| Obscene | 0.9482 | 0.9477 | -0.0005 |
| Threat | 0.8690 | **0.8712** | +0.0022 |
| Insult | 0.9209 | 0.9198 | -0.0012 |
| Identity Hate | 0.8340 | 0.8320 | -0.0020 |

Aggregate metrics:

| Metric | ONNX FP32 | ONNX INT8 |
|---|---:|---:|
| Macro Precision | 0.8431 | **0.8442** |
| Macro Recall | **0.8564** | 0.8524 |
| Macro F1 | **0.8477** | 0.8465 |

### Macro F1 change

```text
0.8477 → 0.8465

Absolute change ≈ -0.0012
```

The evaluation pipeline automatically stores:

```text
evaluation/results/
├── onnx_fp32_metrics.json
├── onnx_int8_metrics.json
├── comparison.csv
└── predictions.csv
```

> The 5,000-row set intentionally oversamples positive/rare labels, so these metrics characterize the controlled comparison set rather than natural production class prevalence.

---

# 🚦 Automated Model Promotion Gate

The project does not promote a model merely because it is faster.

Every candidate must pass configurable quality and performance requirements.

The promotion configuration includes rules such as:

```json
{
  "quality_gates": {
    "max_macro_f1_drop": 0.005,
    "max_category_f1_drop": 0.01,
    "max_macro_recall_drop": 0.01
  },
  "performance_gates": {
    "min_speedup": 1.25,
    "min_size_reduction_percent": 20.0
  }
}
```

The INT8 candidate produced:

```text
MODEL PROMOTION REPORT

Baseline : onnx_fp32
Candidate: onnx_int8

QUALITY GATES

Macro F1       PASS
Macro Recall   PASS

Category F1:

toxic             PASS
severe_toxic      PASS
obscene           PASS
threat            PASS
insult            PASS
identity_hate     PASS

PERFORMANCE GATE

Speedup            PASS   2.40×

MODEL SIZE GATE

Size reduction     PASS   74.84%

OVERALL: PASS
```

The promotion script returns:

```text
exit code 0 → PASS
exit code 1 → FAIL
```

This makes the same model-quality gate usable inside automated CI/CD pipelines.

---

# 🧬 MLflow Experiment Tracking

Model experiments are tracked using **MLflow**.

Tracked information includes:

### Parameters

```text
runtime
base_model
precision
device
classification_threshold
evaluation_rows
```

### Metrics

```text
macro_precision
macro_recall
macro_f1

toxic_precision
toxic_recall
toxic_f1

severe_toxic_precision
severe_toxic_recall
severe_toxic_f1

...

evaluation_ms_per_comment
model_size_mb
```

### Artifacts

```text
evaluation metrics
benchmark results
comparison reports
model artifacts
promotion reports
```

This allows FP32 and INT8 experiments to be compared through the MLflow UI instead of relying on manually recorded results.

---

# 🗂️ MLflow Model Registry

A candidate model is registered only after passing the promotion gate.

The approved INT8 model is registered as:

```text
toxicity-moderation-model
```

with a versioned lifecycle:

```text
Model:
toxicity-moderation-model

Version:
v1

Alias:
candidate

Promotion Gate:
PASSED
```

Conceptually:

```text
Experiment
    ↓
Candidate Model
    ↓
Evaluation
    ↓
Promotion Gate
    ↓
PASS
    ↓
MLflow Model Registry
    ↓
toxicity-moderation-model
    ↓
Versioned Candidate
```

This provides traceability between:

```text
Model Artifact
     ↕
Experiment
     ↕
Metrics
     ↕
Promotion Decision
     ↕
Registered Version
```

---

# 🔥 Load Testing with Locust

The API was load-tested using **Locust** to measure behavior under concurrent traffic.

A 25-user ONNX FP32 test produced:

```text
Requests:       2,178
Failures:       0

Median:         35 ms
Average:        44.94 ms
p95:            100 ms
p99:            220 ms

Current RPS:    ~19.9
```

A prior PyTorch run under the same intended workload produced:

```text
Median:         120 ms
Average:        164.07 ms
p95:            390 ms
p99:            670 ms
RPS:            ~17.6
Failures:       0
```

Observed local end-to-end latency improvements included approximately:

```text
Average latency ↓ ~72.6%
p50 latency     ↓ ~70.8%
p95 latency     ↓ ~74.4%
p99 latency     ↓ ~67.2%
```

These are local load-test observations and depend on hardware, workload, concurrency configuration, container state, and other environmental factors.

---

# 🐘 PostgreSQL Persistence

The platform stores more than predictions.

The database supports:

```text
Users
Moderation Preferences
Predictions
Human Feedback
```

This enables a feedback loop:

```text
Comment
   ↓
Model Prediction
   ↓
Moderation Decision
   ↓
Stored Prediction
   ↓
Human Feedback
   ↓
Future Evaluation / Retraining Data
```

Database schema evolution is managed with **Alembic migrations**.

---

# 👥 Human Feedback Pipeline

Predictions receive persistent IDs.

Users or moderators can submit corrections such as:

```text
Prediction
     ↓
Was the moderation decision correct?
     ↓
Feedback
     ↓
Corrected decision/category
     ↓
Stored in PostgreSQL
```

This creates the foundation for future:

- error analysis
- active learning
- retraining datasets
- drift investigation
- model improvement

The goal is to design the system as a **learning ML platform**, not a static inference endpoint.

---

# 🌐 REST API

FastAPI provides the serving layer.

Core endpoints include:

```text
GET  /health
GET  /ready

POST /users

GET  /users/{user_id}/preferences
PUT  /users/{user_id}/preferences

POST /moderate

GET  /predictions/{prediction_id}

POST /feedback
```

Interactive Swagger documentation is available locally at:

```text
http://127.0.0.1:8000/docs
```

---

# ❤️ Liveness vs Readiness

The platform distinguishes:

```text
/health
```

from:

```text
/ready
```

`/health` answers:

> Is the API process alive?

`/ready` answers:

> Is the inference system actually ready to serve requests?

Example:

```json
{
  "status": "ready",
  "model": "unitary/toxic-bert-onnx",
  "device": "cpu"
}
```

This distinction becomes important when deploying ML services behind orchestration/load-balancing infrastructure.

---

# 🐳 Docker Architecture

The application is containerized with Docker.

Local services include:

```text
Docker Compose
│
├── moderation-api
│
└── moderation-postgres
```

Model artifacts are separated from the application image and mounted read-only at runtime:

```text
Application Image
       +
Model Artifact
       ↓
Running ML Service
```

This prevents every application-code change from requiring the model binary to be rebuilt into the image.

---

# 🔌 Runtime Abstraction

The serving architecture separates the API from the underlying inference runtime.

Conceptually:

```text
FastAPI
   ↓
get_moderation_model()
   ↓
MODEL_RUNTIME
   │
   ├── PyTorch
   │
   └── ONNX
```

Lazy imports prevent the lightweight ONNX production container from requiring PyTorch.

This keeps serving dependencies smaller and separates development/export dependencies from production inference dependencies.

---

# 🧪 Automated Testing

The project includes tests for:

- health checks
- readiness
- safe-comment moderation
- toxic-comment moderation
- request validation
- unknown users
- user creation
- moderation preferences
- policy behavior

Current local test suite:

```text
13 passed
```

Run:

```bash
python -m pytest -v
```

---

# 🔄 Continuous Integration

GitHub Actions runs CI automatically on pushes and pull requests.

The software CI pipeline performs:

```text
Git Push / Pull Request
          ↓
      Checkout
          ↓
    Python 3.13
          ↓
 Install Dependencies
          ↓
       pytest
          ↓
     Docker Build
          ↓
      PASS / FAIL
```

The project deliberately distinguishes between:

### Software CI

```text
code
 ↓
tests
 ↓
container build
```

and:

### Model Release Lifecycle

```text
candidate model
      ↓
evaluation
      ↓
performance benchmark
      ↓
promotion gate
      ↓
MLflow registry
      ↓
deployment candidate
```

Application code and ML models are related, but they do not necessarily have identical release lifecycles.

---

# 🧰 Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.13 |
| NLP Model | Toxic-BERT |
| Transformers | Hugging Face Transformers |
| Baseline Runtime | PyTorch |
| Optimized Runtime | ONNX Runtime |
| Quantization | ONNX Runtime INT8 |
| API | FastAPI |
| Validation | Pydantic |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Migrations | Alembic |
| Containers | Docker / Docker Compose |
| Load Testing | Locust |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| Evaluation | scikit-learn |
| Data Processing | pandas |
| Testing | pytest |
| CI | GitHub Actions |

---

# 📁 Project Structure

```text
ai-moderation-platform/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── alembic/
│   └── versions/
│
├── benchmarks/
│   └── raw/
│
├── config/
│   └── promotion_gate.json
│
├── docker/
│   └── Dockerfile
│
├── evaluation/
│   ├── data/                 # ignored downloaded datasets
│   └── results/
│
├── load_tests/
│   ├── locustfile.py
│   └── test_comments.json
│
├── models/                   # ignored local model artifacts
│   ├── onnx/
│   └── onnx_int8/
│
├── scripts/
│   ├── benchmark.py
│   ├── check_promotion.py
│   ├── compare_quantization.py
│   ├── compare_runtimes.py
│   ├── create_eval_subset.py
│   ├── download_eval_data.py
│   ├── evaluate_models.py
│   ├── export_onnx.py
│   ├── log_mlflow_experiments.py
│   ├── quantize_onnx.py
│   └── register_model.py
│
├── src/
│   ├── api/
│   │   ├── dependencies.py
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── database/
│   │   ├── db.py
│   │   ├── models.py
│   │   └── repository.py
│   │
│   └── model/
│       ├── predictor.py
│       ├── onnx_predictor.py
│       ├── onnx_int8_predictor.py
│       └── policy.py
│
├── tests/
│
├── docker-compose.yml
├── requirements.txt
├── requirements-api.txt
├── requirements-dev.txt
└── pyproject.toml
```

---

# ▶️ Local Development

## Clone

```bash
git clone <repository-url>
cd ai-moderation-platform
```

## Create environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

# 🧪 Run Tests

```bash
python -m pytest -v
```

---

# 🐳 Start Services

Make sure Docker Desktop is running.

Then:

```bash
docker compose up -d
```

Check:

```bash
docker compose ps
```

The expected services are:

```text
moderation-postgres
moderation-api
```

---

# 📊 Run Model Benchmarks

```bash
python -m scripts.benchmark
```

Results are automatically appended to:

```text
benchmarks/raw/model_runtime.csv
```

---

# 📦 Quantize ONNX Model

```bash
python -m scripts.quantize_onnx
```

---

# 🧪 Evaluate Model Quality

Download/build evaluation data:

```bash
python -m scripts.download_eval_data
python -m scripts.create_eval_subset
```

Evaluate:

```bash
python -m scripts.evaluate_models
```

---

# 🚦 Run Promotion Gate

```bash
python -m scripts.check_promotion
```

A passing candidate returns exit code `0`.

A failing candidate returns exit code `1`.

This allows the same command to be used as a future CI/CD quality gate.

---

# 📊 MLflow

Log experiments:

```bash
python -m scripts.log_mlflow_experiments
```

Start MLflow:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

Register an approved model:

```bash
python -m scripts.register_model
```

---

# 🔥 Run Load Tests

Start Locust:

```bash
python -m locust \
  -f load_tests/locustfile.py \
  --host http://127.0.0.1:8000
```

Then open:

```text
http://localhost:8089
```

A reproducible headless run can also be executed with:

```bash
python -m locust \
  -f load_tests/locustfile.py \
  --headless \
  -u 25 \
  -r 5 \
  -t 2m \
  --host http://127.0.0.1:8000 \
  --csv benchmarks/raw/onnx_25_users
```

---

# 🔐 Repository Hygiene

Large/generated/local artifacts are intentionally excluded from Git:

```text
models/
evaluation/data/
mlflow.db
mlruns/
mlartifacts/
.env
.venv/
__pycache__/
```

This prevents:

- model binaries from bloating Git history
- evaluation datasets from being accidentally committed
- local MLflow state from entering source control
- secrets from being exposed
- Python-generated bytecode from polluting commits

---

# 💡 Key Engineering Lessons Demonstrated

This project intentionally focuses on the parts of ML engineering that happen **after a model exists**.

### 1. Model accuracy is only one part of production ML

A production model also needs:

```text
latency
throughput
memory/storage efficiency
reliability
observability
versioning
testing
deployment safety
```

### 2. Faster does not automatically mean better

INT8 was substantially faster and smaller, but it was not promoted until its quality was evaluated.

```text
Optimize
   ↓
Measure
   ↓
Evaluate quality
   ↓
Apply promotion gate
   ↓
Register
```

### 3. Tail latency matters

Average latency alone can hide poor user experiences.

Therefore the project records:

```text
p50
p95
p99
```

rather than relying only on averages.

### 4. ML and product policy should be separate

The classifier predicts probabilities.

The policy engine decides how those probabilities affect a particular user.

```text
ML prediction ≠ product decision
```

### 5. Model releases need gates

A new model should not reach deployment merely because training/export completed successfully.

It should satisfy measurable requirements first.

### 6. Reproducibility matters

Benchmarks and evaluations write results to machine-readable artifacts rather than relying solely on screenshots or manually copied numbers.

---

# 🔮 Future Scope

The architecture can be extended into a larger social-media moderation platform.

Potential additions include:

### Real-Time Social Media Integration

Moderate:

- post comments
- story replies
- live-stream chats
- community messages

### Streaming Architecture

For very high traffic:

```text
Social Platform
      ↓
Kafka / Event Stream
      ↓
Moderation Workers
      ↓
ONNX Runtime
      ↓
Moderation Decision
```

### Monitoring

Add:

```text
Prometheus
    ↓
Grafana
```

for:

- request latency
- inference latency
- error rates
- throughput
- category distributions
- model confidence
- moderation decision rates

### Drift Detection

Monitor changes in:

```text
language patterns
toxicity distributions
model confidence
user feedback
false positives
false negatives
```

### Automated Retraining

```text
Production Predictions
        ↓
Human Feedback
        ↓
Validated Dataset
        ↓
Retraining Pipeline
        ↓
Candidate Model
        ↓
Evaluation
        ↓
Promotion Gate
        ↓
Registry
        ↓
Deployment
```

### Champion / Challenger Deployment

Future model versions could be evaluated against the current production model before receiving a production alias.

---

# 🌟 What Makes This Project Different?

This is intentionally **not**:

```text
Load model
→ call predict()
→ put it behind Flask
→ done
```

Instead, it treats ML as a complete production system:

```text
                    DATA
                     ↓
                 ML MODEL
                     ↓
                  SERVING
                     ↓
                PERSISTENCE
                     ↓
                 FEEDBACK
                     ↓
               EVALUATION
                     ↓
               OPTIMIZATION
                     ↓
                BENCHMARKING
                     ↓
             EXPERIMENT TRACKING
                     ↓
               QUALITY GATES
                     ↓
                MODEL REGISTRY
                     ↓
                    CI/CD
                     ↓
                 MONITORING
                     ↓
                 RETRAINING
```

The central engineering question is not simply:

> **“Can the model classify toxic text?”**

It is:

> **“Can we build a measurable, reproducible, personalized, performant, and safely upgradable ML system around that model?”**

That is the purpose of this project.

---

# 📌 Current Project Status

Implemented:

- [x] Toxic-BERT inference
- [x] Multi-label toxicity classification
- [x] Personalized moderation preferences
- [x] Policy engine
- [x] FastAPI serving
- [x] PostgreSQL persistence
- [x] SQLAlchemy ORM
- [x] Alembic migrations
- [x] Prediction history
- [x] Human feedback collection
- [x] Automated tests
- [x] Dockerized application
- [x] Locust load testing
- [x] PyTorch baseline benchmarking
- [x] ONNX export
- [x] PyTorch/ONNX prediction comparison
- [x] ONNX Runtime optimization
- [x] INT8 quantization
- [x] Reproducible performance benchmarking
- [x] 5,000-comment quality evaluation
- [x] Precision / Recall / F1 evaluation
- [x] Automated model promotion gate
- [x] MLflow experiment tracking
- [x] MLflow Model Registry
- [x] Registered/versioned candidate model
- [x] GitHub Actions CI

In progress / future:

- [ ] Production monitoring
- [ ] Prometheus metrics
- [ ] Grafana dashboards
- [ ] Drift detection
- [ ] Automated retraining
- [ ] Model-release CI/CD
- [ ] Cloud deployment
- [ ] Champion/challenger rollout

---

# ⚠️ Responsible Use

Automated toxicity classifiers are imperfect.

Language is contextual, culturally dependent, and constantly evolving. Sarcasm, reclaimed language, slang, quotations, and friendly banter can all produce unexpected predictions.

For example, a phrase intended as praise can sometimes receive a high toxicity score when context is missing.

For this reason, this project intentionally includes:

- configurable moderation policy
- human feedback
- per-category evaluation
- model-quality gates
- model versioning
- future monitoring/retraining hooks

High-impact moderation systems should not treat a model probability as unquestionable ground truth.

---

# 📄 License

Add the license appropriate for your intended use.

---

# 👨‍💻 Author

Built as an end-to-end **Machine Learning Engineering / MLOps portfolio project** exploring how modern NLP models can be optimized, evaluated, versioned, served, monitored, and safely promoted in a production-oriented architecture.
