from typing import Literal, Optional

from app.routes.schemas.base import BaseSchema
from app.routes.schemas.conversation import Content, MessageOutput, type_model_name
from pydantic import Field


class MessageInputWithoutMessageid(BaseSchema):
    content: list[Content]
    model: type_model_name


class ChatInputWithoutBotId(BaseSchema):
    conversation_id: str | None = Field(
        None,
        description="""Unique conversation id. 
        If not provided, new conversation will be generated.
        Please note that the id must be unique across the application.
        We recommend to use auto-generated id (i.e. give None to this field)""",
    )
    message: MessageInputWithoutMessageid


class ChatOutputWithoutBotId(BaseSchema):
    conversation_id: str
    message: MessageOutput
    create_time: float


class MessageRequestedResponse(BaseSchema):
    conversation_id: str
    message_id: str
