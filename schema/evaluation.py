from typing import Dict, Optional, Union, cast

from pydantic import (
    BaseModel,
    ValidationError,
    root_validator,  # type: ignore
    validator,  # type: ignore
)
from utils.constants import (
    ALLOWED_CLASSIFICATION_METRICS,
    ALLOWED_REGRESSION_METRICS,
    BALANCED_BOOL_MAP,
    CLASSIFICATION,
    REGRESSION,
)


class EvaluationSchema(BaseModel):
    """pydantic schema for JSON Validations"""

    problem_type: str
    dataset_balanced: bool = False
    metric: Optional[str] = None

    @validator("problem_type")
    @classmethod
    def problem_type_validations(cls, problem_type: str) -> str:
        """validate problem_type"""
        if problem_type not in [CLASSIFICATION, REGRESSION]:
            raise ValueError("Problem type not supported")

        return problem_type

    @root_validator  # type: ignore
    @classmethod
    def final_validations(
        cls, values: Dict[str, Union[str, bool, None]]
    ) -> Dict[str, Union[str, bool, None]]:
        """final validations"""
        metric = values.get("metric")
        if not metric:
            return values

        problem_type = values.get("problem_type")
        if problem_type == CLASSIFICATION:
            dataset_balanced = cast(bool, values.get("dataset_balanced", False))
            balanced_identifier = BALANCED_BOOL_MAP[dataset_balanced]
            if metric not in ALLOWED_CLASSIFICATION_METRICS[balanced_identifier]:
                raise ValueError("Metric not supported")
        if problem_type == REGRESSION:
            if metric not in ALLOWED_REGRESSION_METRICS:
                raise ValueError("Metric not supported")

        return values


def validate(data: Dict[str, Union[str, bool, None]]) -> EvaluationSchema:
    """Validate data with pydantic schema"""
    try:
        valid_data = EvaluationSchema(**data)  # type: ignore
        return valid_data
    except ValidationError as exc:
        raise exc
