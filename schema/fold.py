from typing import Dict, Union

from pydantic import (
    BaseModel,
    ValidationError,
    validator,  # type: ignore
)
from utils.constants import CLASSIFICATION, REGRESSION


class FoldSchema(BaseModel):
    """pydantic schema for JSON Validations"""

    problem_type: str
    dataset_balanced: bool = False

    @validator("problem_type")
    @classmethod
    def problem_type_validations(cls, problem_type: str) -> str:
        """validate problem_type"""
        if problem_type not in [CLASSIFICATION, REGRESSION]:
            raise ValueError("Problem type not supported")

        return problem_type


def validate(data: Dict[str, Union[str, bool, None]]) -> FoldSchema:
    """Validate data with pydantic schema"""
    try:
        valid_data = FoldSchema(**data)  # type: ignore
        return valid_data
    except ValidationError as exc:
        raise exc
