import json
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from src.models import Observation, Action, Reward


@dataclass
class EpisodeState:
    dataset: pd.DataFrame
    original_dataset: pd.DataFrame
    task_id: str
    step_count: int = 0
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    missing_values_identified: Dict[str, int] = field(default_factory=dict)
    duplicates_found: int = 0
    quality_issues: List[str] = field(default_factory=list)


class DataCleaningEnv:
    def __init__(self):
        self.current_episode: Optional[EpisodeState] = None
        self.max_steps = 50
        self.dataset_templates = self._create_dataset_templates()

    def _create_dataset_templates(self) -> Dict[str, pd.DataFrame]:
        templates = {}
        
        np.random.seed(42)
        
        customer_data = pd.DataFrame({
            'customer_id': list(range(1, 101)),
            'name': [f'Customer_{i}' if random.random() > 0.15 else None for i in range(100)],
            'email': [f'customer{i}@email.com' if random.random() > 0.1 else None for i in range(100)],
            'age': [random.randint(18, 80) if random.random() > 0.08 else None for _ in range(100)],
            'signup_date': pd.date_range('2020-01-01', periods=100, freq='D').astype(str),
            'revenue': [round(random.uniform(100, 5000), 2) if random.random() > 0.12 else None for _ in range(100)],
            'country': [random.choice(['US', 'UK', 'CA', 'DE', None]) for _ in range(100)]
        })
        customer_data = pd.concat([customer_data, customer_data.iloc[:15]])
        templates['customer_database'] = customer_data.reset_index(drop=True)
        
        sales_data = pd.DataFrame({
            'order_id': list(range(1000, 1200)),
            'product': [random.choice(['Laptop', 'Phone', 'Tablet', None]) for _ in range(200)],
            'quantity': [random.randint(1, 10) if random.random() > 0.1 else None for _ in range(200)],
            'price': [random.uniform(50, 2000) if random.random() > 0.08 else None for _ in range(200)],
            'sale_date': pd.date_range('2023-01-01', periods=200, freq='12h').astype(str),
            'status': [random.choice(['completed', 'pending', 'cancelled', None]) for _ in range(200)]
        })
        sales_data = pd.concat([sales_data, sales_data.iloc[:20]])
        templates['sales_records'] = sales_data.reset_index(drop=True)
        
        employee_data = pd.DataFrame({
            'emp_id': [f'EMP{i:04d}' for i in range(1, 151)],
            'salary': [random.randint(50000, 150000) if random.random() > 0.1 else None for _ in range(150)],
            'department': [random.choice(['Engineering', 'Sales', 'HR', 'Finance', None]) for _ in range(150)],
            'hire_date': [pd.Timestamp('2015-01-01') + pd.Timedelta(days=i*2) for i in range(150)],
            'performance_score': [random.uniform(1, 5) if random.random() > 0.12 else None for _ in range(150)]
        })
        employee_data = pd.concat([employee_data, employee_data.iloc[:10]])
        templates['employee_data'] = employee_data.reset_index(drop=True)
        
        return templates

    def reset(self, task_id: str = 'task_missing_values') -> Observation:
        selected_template = random.choice(list(self.dataset_templates.values()))
        dataset = selected_template.copy()
        
        self.current_episode = EpisodeState(
            dataset=dataset,
            original_dataset=dataset.copy(),
            task_id=task_id,
            step_count=0
        )
        
        return self._get_observation()

    def _get_observation(self) -> Observation:
        if not self.current_episode:
            raise RuntimeError("Episode not initialized. Call reset() first.")
        
        episode = self.current_episode
        missing_values = episode.dataset.isnull().sum().to_dict()
        
        return Observation(
            dataset_shape=tuple(episode.dataset.shape),
            column_names=list(episode.dataset.columns),
            data_types={col: str(dtype) for col, dtype in episode.dataset.dtypes.items()},
            missing_values=missing_values,
            current_state=self._describe_state(),
            task_id=episode.task_id,
            step_count=episode.step_count,
            episode_progress=self._get_progress_summary()
        )

    def _describe_state(self) -> str:
        if not self.current_episode:
            return "No active episode"
        
        episode = self.current_episode
        rows, cols = episode.dataset.shape
        missing_count = episode.dataset.isnull().sum().sum()
        dup_count = episode.dataset.duplicated(subset=None, keep=False).sum()
        
        return f"Dataset({rows} rows, {cols} cols): {missing_count} missing values, {dup_count} potential duplicates"

    def _get_progress_summary(self) -> str:
        if not self.current_episode:
            return "No progress"
        
        episode = self.current_episode
        if not episode.actions_taken:
            return "No actions taken yet"
        
        return f"Completed {len(episode.actions_taken)} action(s): {', '.join([a.get('action_type', '?') for a in episode.actions_taken[-3:]])}"

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if not self.current_episode:
            raise RuntimeError("Episode not initialized. Call reset() first.")
        
        episode = self.current_episode
        episode.step_count += 1
        
        episode.actions_taken.append({
            'action_type': action.action_type,
            'target_columns': action.target_columns,
            'parameters': action.parameters,
            'step': episode.step_count
        })
        
        reward, info = self._process_action(action)
        
        done = episode.step_count >= self.max_steps
        
        observation = self._get_observation()
        
        return observation, reward, done, info

    def _process_action(self, action: Action) -> Tuple[Reward, Dict[str, Any]]:
        if not self.current_episode:
            raise RuntimeError("No active episode")
        
        episode = self.current_episode
        components = {}
        messages = []
        
        if action.action_type == 'analyze':
            components['analysis'] = self._perform_analysis(action.target_columns)
            messages.append(f"Analyzed {len(action.target_columns)} columns")
        
        elif action.action_type == 'impute':
            impute_reward = self._perform_imputation(action.target_columns, action.parameters)
            components['imputation'] = impute_reward
            messages.append("Imputation applied")
        
        elif action.action_type == 'deduplicate':
            dedup_reward = self._perform_deduplication(action.parameters)
            components['deduplication'] = dedup_reward
            messages.append("Deduplication executed")
        
        elif action.action_type == 'validate':
            validation_reward = self._perform_validation(action.target_columns, action.parameters)
            components['validation'] = validation_reward
            messages.append("Validation performed")
        
        elif action.action_type == 'report_findings':
            report_reward = self._generate_report(action.parameters)
            components['reporting'] = report_reward
            messages.append("Report generated")
        
        else:
            components['invalid_action'] = 0.0
            messages.append(f"Unknown action type: {action.action_type}")
        
        if not components:
            total_reward = 0.0
        else:
            total_reward = sum(components.values()) / len(components) if components else 0.0
        
        total_reward = min(1.0, max(0.0, total_reward))
        
        info = {
            'action_type': action.action_type,
            'reasoning': action.reasoning,
            'components': components,
            'messages': messages
        }
        
        return Reward(
            value=total_reward,
            components=components,
            message="; ".join(messages)
        ), info

    def _perform_analysis(self, columns: List[str]) -> float:
        if not self.current_episode:
            return 0.0
        
        dataset = self.current_episode.dataset
        valid_cols = [c for c in columns if c in dataset.columns]
        
        if not valid_cols:
            return 0.0
        
        reward = 0.0
        for col in valid_cols:
            missing_pct = dataset[col].isnull().sum() / len(dataset)
            if missing_pct > 0:
                reward += 0.3
            
            if dataset[col].dtype == 'object':
                reward += 0.1
        
        return min(1.0, reward / len(valid_cols))

    def _perform_imputation(self, columns: List[str], params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        
        episode = self.current_episode
        dataset = episode.dataset
        valid_cols = [c for c in columns if c in dataset.columns]
        
        if not valid_cols:
            return 0.0
        
        reward = 0.0
        method = params.get('method', 'mean')
        
        for col in valid_cols:
            if dataset[col].isnull().sum() == 0:
                reward += 0.1
            elif method in ['mean', 'median'] and dataset[col].dtype in [np.float64, np.int64]:
                if method == 'mean':
                    value = dataset[col].mean()
                else:
                    value = dataset[col].median()
                
                if pd.notna(value):
                    dataset[col].fillna(value, inplace=True)
                    missing_before = dataset[col].isnull().sum()
                    reward += 0.4 if missing_before == 0 else 0.2
            elif method == 'forward_fill':
                dataset[col].fillna(method='ffill', inplace=True)
                reward += 0.35
        
        return min(1.0, reward / len(valid_cols)) if valid_cols else 0.0

    def _perform_deduplication(self, params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        
        episode = self.current_episode
        dataset = episode.dataset
        
        dup_count_before = dataset.duplicated(subset=None, keep=False).sum()
        
        if dup_count_before == 0:
            return 0.5
        
        subset = params.get('subset', None)
        keep = params.get('keep', 'first')
        
        if subset and all(c in dataset.columns for c in subset):
            dataset.drop_duplicates(subset=subset, keep=keep, inplace=True)
        else:
            dataset.drop_duplicates(keep=keep, inplace=True)
        
        dup_count_after = dataset.duplicated(subset=None, keep=False).sum()
        
        if dup_count_after == 0:
            return 0.9
        else:
            return max(0.4, 1.0 - (dup_count_after / dup_count_before))

    def _perform_validation(self, columns: List[str], params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        
        dataset = self.current_episode.dataset
        valid_cols = [c for c in columns if c in dataset.columns]
        
        if not valid_cols:
            return 0.0
        
        reward = 0.0
        
        for col in valid_cols:
            validation_type = params.get(f'{col}_type', 'exists')
            
            if validation_type == 'exists':
                if dataset[col].isnull().sum() == 0:
                    reward += 0.5
                else:
                    reward += 0.2
            
            elif validation_type == 'numeric':
                if dataset[col].dtype in [np.float64, np.int64]:
                    reward += 0.5
                else:
                    reward += 0.1
            
            elif validation_type == 'range':
                min_val = params.get(f'{col}_min', dataset[col].min())
                max_val = params.get(f'{col}_max', dataset[col].max())
                
                in_range = ((dataset[col] >= min_val) & (dataset[col] <= max_val)).sum()
                reward += (in_range / len(dataset)) * 0.5
        
        return min(1.0, reward / len(valid_cols)) if valid_cols else 0.0

    def _generate_report(self, params: Dict[str, Any]) -> float:
        if not self.current_episode:
            return 0.0
        
        episode = self.current_episode
        dataset = episode.dataset
        original = episode.original_dataset
        
        reward = 0.0
        
        if params.get('include_summary', False):
            reward += 0.2
        
        if params.get('include_quality_score', False):
            reward += 0.2
        
        if params.get('include_recommendations', False):
            reward += 0.2
        
        rows_original = len(original)
        rows_cleaned = len(dataset)
        cols_original = len(original.columns)
        cols_cleaned = len(dataset.columns)
        
        if rows_cleaned < rows_original:
            reward += 0.15
        
        if dataset.isnull().sum().sum() < original.isnull().sum().sum():
            reward += 0.25
        
        return min(1.0, reward)

    def state(self) -> Dict[str, Any]:
        if not self.current_episode:
            return {'error': 'No active episode'}
        
        episode = self.current_episode
        
        return {
            'dataset_shape': tuple(episode.dataset.shape),
            'missing_values_count': int(episode.dataset.isnull().sum().sum()),
            'duplicates_count': int(episode.dataset.duplicated(subset=None, keep=False).sum()),
            'columns': list(episode.dataset.columns),
            'step': episode.step_count,
            'task_id': episode.task_id,
            'actions': len(episode.actions_taken)
        }
