from pydantic import BaseModel


class AnswerPost(BaseModel):
    rows: list[dict[str, int |
                         bool|
                         str |
                         dict[str, list[str]]]]