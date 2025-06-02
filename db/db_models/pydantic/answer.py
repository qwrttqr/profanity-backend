from pydantic import BaseModel


class AnswerPost(BaseModel):
    rows: list[dict[str, int | bool | dict[str, list[str]]]]