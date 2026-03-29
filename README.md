---
title: scaler-ml-submission
emoji: "🐳"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Data Cleaning Environment

A real-world data cleaning and validation environment for training and evaluating AI agents. Built with OpenEnv specification compliance.

## Overview

This environment simulates the practical task of data cleaning and quality assurance that data engineers and analysts perform daily. Agents must identify data quality issues (missing values, duplicates, type mismatches) and apply appropriate cleaning strategies to improve dataset quality.

### Problem Domain

Real organizations face constant challenges with data quality:
- Customer relationship systems accumulate incomplete records
- Sales databases develop duplicates during bulk imports
- Employee records have inconsistent formats and missing information
- Data pipelines require validation before processing

This environment enables training agents to autonomously handle these realistic scenarios.

## Environment Design

### Observation Space

```
Observation {
  dataset_shape: tuple[int, int]        - Current dataset dimensions (rows, columns)
  column_names: List[str]               - Names of all columns
  data_types: Dict[str, str]            - Data type for each column
  missing_values: Dict[str, int]        - Count of missing values per column
  current_state: str                    - Human-readable state description
  task_id: str                          - Current task identifier
  step_count: int                       - Current step within episode
  episode_progress: str                 - Summary of actions taken
}
```

### Action Space

```
Action {
  action_type: str                      - One of: analyze, impute, deduplicate, validate, report_findings
  target_columns: List[str]             - Columns to process
  parameters: Dict[str, Any]            - Method-specific parameters
  reasoning: str                        - Explanation of the action
}
```

**Action Types:**
- `analyze` - Examine data structure and identify quality issues
- `impute` - Fill missing values with mean, median, or forward fill
- `deduplicate` - Remove duplicate records
- `validate` - Check data against validation rules
- `report_findings` - Generate dataset quality summary

### Reward Signal

Rewards range from 0.0 to 1.0 and provide incremental feedback:
- Completing meaningful cleaning actions: +0.2 to +0.5
- Reducing missing values: proportional to reduction
- Removing duplicates: proportional to deduplication rate
- Proper validation execution: +0.1 to +0.3
- High-quality final state: +0.2 bonus

## Tasks

### Task 1: Missing Values Detection and Strategy (Easy)

**Objective:** Identify missing values and apply an appropriate imputation strategy.

**Difficulty:** Easy

**Success Criteria:**
- Detect columns with missing data
- Apply appropriate imputation method
- Reduce missing value count below 10% of dataset

**Baseline:** 0.65-0.75

### Task 2: Duplicate Detection and Removal (Medium)

**Objective:** Identify and remove duplicate records from dataset.

**Difficulty:** Medium

**Success Criteria:**
- Identify duplicate rows
- Apply deduplication strategy
- Remove 90%+ of detected duplicates
- Maintain data integrity

**Baseline:** 0.55-0.70

### Task 3: Complex Data Quality Validation (Hard)

**Objective:** Comprehensive quality assessment with type validation, domain rules, and recommendation generation.

**Difficulty:** Hard

**Success Criteria:**
- Analyze multiple dimensions (types, ranges, duplicates, missing values)
- Apply diverse cleaning strategies
- Validate improvements
- Generate actionable quality report
- Achieve improvements across all dimensions

**Baseline:** 0.50-0.65

## Grader Functions

Each task includes a deterministic grader that scores agent performance:

```python
score = grader.grade(episode_state: EpisodeState) -> float
```

Graders evaluate:
- Reduction in data quality issues (missing values, duplicates)
- Diversity and correctness of actions taken
- Proper use of validation and analysis
- Quality of final state vs. initial state

## Setup Instructions

### Requirements
- Python 3.10+
- pip or conda

### Installation

```bash
git clone https://github.com/YOUR_ORG/data-cleaning-env.git
cd data-cleaning-env

pip install -r requirements.txt
pip install -e .
```

### Environment Variables

For running inference:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="hf_your_token_here"
export OPENAI_API_KEY="sk_your_key_here"
```

## Usage

### Basic Environment Use

```python
from src.environment import DataCleaningEnv
from src.models import Action

env = DataCleaningEnv()

observation = env.reset(task_id="task_missing_values")

action = Action(
    action_type="analyze",
    target_columns=["customer_id", "email"],
    parameters={},
    reasoning="Initial analysis of data structure"
)

observation, reward, done, info = env.step(action)

state = env.state()
```

### Running Server

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Running Baseline Inference

```bash
python inference.py
```

Expected output:
```
============================================================
Data Cleaning Environment - Baseline Inference
============================================================

Running task_missing_values
Initial state: Dataset(100 rows, 6 cols): 15 missing values, 8 potential duplicates
  Step 1: analyze -> reward 0.150
  Step 2: impute -> reward 0.450
  Step 3: validate -> reward 0.200
Task task_missing_values final score: 0.720

Running task_duplicate_handling
Initial state: Dataset(115 rows, 5 cols): 12 missing values, 20 potential duplicates
  Step 1: analyze -> reward 0.120
  Step 2: deduplicate -> reward 0.550
Task task_duplicate_handling final score: 0.680

Running task_complex_validation
Initial state: Dataset(150 rows, 5 cols): 18 missing values, 10 potential duplicates
  Step 1: analyze -> reward 0.140
  Step 2: impute -> reward 0.350
  Step 3: deduplicate -> reward 0.400
  Step 4: validate -> reward 0.300
  Step 5: report_findings -> reward 0.200
Task task_complex_validation final score: 0.570

============================================================
Baseline Scores
============================================================
task_missing_values: 0.720
task_duplicate_handling: 0.680
task_complex_validation: 0.570

Average score: 0.657
```

## Validation

### OpenEnv Compliance

Verify spec compliance:

```bash
pip install openenv-core
openenv validate
```

### Docker Build

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4" \
  -e HF_TOKEN="hf_token" \
  -e OPENAI_API_KEY="sk_key" \
  data-cleaning-env
```

## Deployment

### Hugging Face Spaces

1. Create repository on Hugging Face
2. Push code to main branch
3. Create Space from repository
4. Set environment variables in Space settings
5. Space will auto-deploy with Dockerfile

Space URL format: `https://your-username-<space-name>.hf.space`

## File Structure

```
data-cleaning-env/
├── src/
│   ├── __init__.py
│   ├── models.py           - Pydantic models for Observation, Action, Reward
│   ├── environment.py      - Main environment class
│   └── graders.py          - Task grader implementations
├── server/
│   └── app.py              - FastAPI server for Space deployment
├── inference.py            - Baseline inference script with OpenAI API
├── openenv.yaml            - OpenEnv specification
├── setup.py                - Package setup
├── requirements.txt        - Dependencies
├── Dockerfile              - Container configuration
└── README.md               - This file
```

## Baseline Scores

The baseline inference script uses GPT-4 with:
- Temperature: 0.3 (deterministic)
- Max tokens: 500 per step
- Max steps: 20 per task

Reproducible baseline scores:
- Task 1 (Missing Values): ~0.72
- Task 2 (Duplicate Handling): ~0.68
- Task 3 (Complex Validation): ~0.57
- **Average: ~0.66**

## Implementation Notes

- All random seeds are fixed for reproducibility
- Datasets are procedurally generated with consistent quality metrics
- Graders use deterministic scoring functions
- Episode max length: 50 steps (configurable)
- Action validation happens server-side

## Contact & Support

For issues or questions, please open an issue on the GitHub repository.

## License

Open source - available for training and evaluation purposes.