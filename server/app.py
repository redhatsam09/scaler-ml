from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from src.environment import DataCleaningEnv
from src.models import Action, Observation
from src.graders import MissingValuesGrader, DuplicateHandlingGrader, ComplexValidationGrader

app = FastAPI(title="Data Cleaning Environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = DataCleaningEnv()

environments: Dict[str, DataCleaningEnv] = {}


class ResetRequest(BaseModel):
    task_id: str = "task_missing_values"


class StepRequest(BaseModel):
    action: Dict[str, Any]


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    task_id: str
    step: int


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    state: Dict[str, Any]


class GradeResponse(BaseModel):
    task_id: str
    score: float


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "name": "Data Cleaning Environment",
        "version": "1.0.0",
        "tasks": ["task_missing_values", "task_duplicate_handling", "task_complex_validation"]
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(request: Optional[ResetRequest] = None):
    try:
        task_id = request.task_id if request else "task_missing_values"
        observation = env.reset(task_id=task_id)
        
        return ResetResponse(
            observation={
                "dataset_shape": observation.dataset_shape,
                "column_names": observation.column_names,
                "data_types": observation.data_types,
                "missing_values": observation.missing_values,
                "current_state": observation.current_state,
                "task_id": observation.task_id,
                "step_count": observation.step_count,
                "episode_progress": observation.episode_progress,
            },
            task_id=observation.task_id,
            step=observation.step_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    try:
        if env.current_episode is None:
            raise ValueError("Environment not initialized. Call /reset first.")
        
        action = Action(**request.action)
        
        observation, reward, done, info = env.step(action)
        
        return StepResponse(
            observation={
                "dataset_shape": observation.dataset_shape,
                "column_names": observation.column_names,
                "data_types": observation.data_types,
                "missing_values": observation.missing_values,
                "current_state": observation.current_state,
                "task_id": observation.task_id,
                "step_count": observation.step_count,
                "episode_progress": observation.episode_progress,
            },
            reward=reward.value,
            done=done,
            info=info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/state", response_model=StateResponse)
async def state():
    try:
        return StateResponse(state=env.state())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grade", response_model=GradeResponse)
async def grade(task_id: str = "task_missing_values"):
    try:
        if env.current_episode is None or env.current_episode.task_id != task_id:
            raise ValueError("Episode not matching requested task")
        
        graders = {
            "task_missing_values": MissingValuesGrader,
            "task_duplicate_handling": DuplicateHandlingGrader,
            "task_complex_validation": ComplexValidationGrader,
        }
        
        grader_class = graders.get(task_id)
        if not grader_class:
            raise ValueError(f"Unknown task: {task_id}")
        
        score = grader_class.grade(env.current_episode)
        
        return GradeResponse(task_id=task_id, score=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
