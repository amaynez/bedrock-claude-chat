import sys
import unittest

sys.path.append(".")
from app.repositories.conversation import (
    RecordNotFoundError,
    _compose_conv_id,
    _get_table_client,
    change_conversation_title,
    delete_conversation_by_id,
    delete_conversation_by_user_id,
    find_conversation_by_id,
    find_conversation_by_user_id,
    store_conversation,
)
from app.repositories.custom_bot import (
    delete_bot_by_id,
    find_bot_by_id,
    find_bot_by_user_id,
    store_bot,
)
from app.repositories.model import (
    BotModel,
    ContentModel,
    ConversationModel,
    MessageModel,
)
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError


class TestRowLevelAccess(unittest.TestCase):
    def setUp(self) -> None:
        self.conversation_user_1 = ConversationModel(
            id="1",
            create_time=1627984879.9,
            title="Test Conversation",
            messages=[
                MessageModel(
                    id="1",
                    role="user",
                    content=ContentModel(content_type="text", body="Hello"),
                    model="model",
                    create_time=1627984879.9,
                )
            ],
        )
        store_conversation("user1", self.conversation_user_1)

        self.conversation_user_2 = ConversationModel(
            id="2",
            create_time=1627984879.9,
            title="Test Conversation",
            messages=[
                MessageModel(
                    id="1",
                    role="user",
                    content=ContentModel(content_type="text", body="Hello"),
                    model="model",
                    create_time=1627984879.9,
                )
            ],
        )
        store_conversation("user2", self.conversation_user_2)

    def test_find_conversation_by_user_id(self):
        # Create table client for user1
        table = _get_table_client("user1")

        table.query(
            KeyConditionExpression=Key("PK").eq(
                _compose_conv_id("user1", self.conversation_user_1.id)
            )
        )

        with self.assertRaises(ClientError):
            # Raise `AccessDeniedException` because user1 cannot access user2's data
            table.query(
                KeyConditionExpression=Key("PK").eq(
                    _compose_conv_id("user2", self.conversation_user_2.id)
                )
            )

    def test_find_conversation_by_id(self):
        # Create table client for user1
        table = _get_table_client("user1")

        table.query(
            IndexName="SKIndex",
            KeyConditionExpression=Key("SK").eq(
                _compose_conv_id("user1", self.conversation_user_1.id)
            ),
        )
        with self.assertRaises(ClientError):
            # Raise `AccessDeniedException` because user1 cannot access user2's data
            table.query(
                IndexName="SKIndex",
                KeyConditionExpression=Key("SK").eq(
                    _compose_conv_id("user2", self.conversation_user_2.id)
                ),
            )

    def tearDown(self) -> None:
        delete_conversation_by_user_id("user1")
        delete_conversation_by_user_id("user2")


class TestConversationRepository(unittest.TestCase):
    def test_store_and_find_conversation(self):
        conversation = ConversationModel(
            id="1",
            create_time=1627984879.9,
            title="Test Conversation",
            message_map={
                "a": MessageModel(
                    role="user",
                    content=ContentModel(content_type="text", body="Hello"),
                    model="model",
                    children=["x", "y"],
                    parent="z",
                    create_time=1627984879.9,
                )
            },
            last_message_id="x",
            bot_id="1",
        )

        # Test storing conversation
        response = store_conversation("user", conversation)
        self.assertIsNotNone(response)

        # Test finding conversation by user_id
        conversations = find_conversation_by_user_id(user_id="user")
        self.assertEqual(len(conversations), 1)

        # Test finding conversation by id
        found_conversation = find_conversation_by_id(
            user_id="user", conversation_id="1"
        )
        self.assertEqual(found_conversation.id, "1")
        message_map = found_conversation.message_map
        # Assert whether the message map is correctly reconstructed
        self.assertEqual(message_map["a"].role, "user")
        self.assertEqual(message_map["a"].content.content_type, "text")
        self.assertEqual(message_map["a"].content.body, "Hello")
        self.assertEqual(message_map["a"].model, "model")
        self.assertEqual(message_map["a"].children, ["x", "y"])
        self.assertEqual(message_map["a"].parent, "z")
        self.assertEqual(message_map["a"].create_time, 1627984879.9)
        self.assertEqual(found_conversation.last_message_id, "x")
        self.assertEqual(found_conversation.bot_id, "1")

        # Test update title
        response = change_conversation_title(
            user_id="user",
            conversation_id="1",
            new_title="Updated title",
        )
        found_conversation = find_conversation_by_id(
            user_id="user", conversation_id="1"
        )
        self.assertEqual(found_conversation.title, "Updated title")

        # Test deleting conversation by id
        delete_conversation_by_id(user_id="user", conversation_id="1")
        with self.assertRaises(RecordNotFoundError):
            find_conversation_by_id("user", "1")

        response = store_conversation(user_id="user", conversation=conversation)

        # Test deleting conversation by user_id
        delete_conversation_by_user_id(user_id="user")
        conversations = find_conversation_by_user_id(user_id="user")
        self.assertEqual(len(conversations), 0)


class TestConversationBotRepository(unittest.TestCase):
    def setUp(self) -> None:
        conversation1 = ConversationModel(
            id="1",
            create_time=1627984879.9,
            title="Test Conversation",
            message_map={
                "a": MessageModel(
                    role="user",
                    content=ContentModel(content_type="text", body="Hello"),
                    model="model",
                    children=["x", "y"],
                    parent="z",
                    create_time=1627984879.9,
                )
            },
            last_message_id="x",
            bot_id=None,
        )
        conversation2 = ConversationModel(
            id="2",
            create_time=1627984879.9,
            title="Test Conversation",
            message_map={
                "a": MessageModel(
                    role="user",
                    content=ContentModel(content_type="text", body="Hello"),
                    model="model",
                    children=["x", "y"],
                    parent="z",
                    create_time=1627984879.9,
                )
            },
            last_message_id="x",
            bot_id="1",
        )
        bot1 = BotModel(
            id="1",
            title="Test Bot",
            instruction="Test Bot Prompt",
            description="Test Bot Description",
            create_time=1627984879.9,
            last_used_time=1627984879.9,
        )
        bot2 = BotModel(
            id="2",
            title="Test Bot",
            instruction="Test Bot Prompt",
            description="Test Bot Description",
            create_time=1627984879.9,
            last_used_time=1627984879.9,
        )

        store_conversation("user", conversation1)
        store_bot("user", bot1)
        store_bot("user", bot2)
        store_conversation("user", conversation2)

    def test_only_conversation_is_fetched(self):
        conversations = find_conversation_by_user_id("user")
        self.assertEqual(len(conversations), 2)

    def test_only_bot_is_fetched(self):
        bots = find_bot_by_user_id("user")
        self.assertEqual(len(bots), 2)

    def test_last_bot_used_updated(self):
        # Update existing conversation
        conversation = ConversationModel(
            id="2",
            create_time=1627984879.9,
            title="Test Conversation",
            message_map={
                "a": MessageModel(
                    role="user",
                    content=ContentModel(content_type="text", body="Hello"),
                    model="model",
                    children=["x", "y"],
                    parent="z",
                    create_time=1627984879.9,
                )
            },
            last_message_id="x",
            bot_id="1",
        )
        bot_before = find_bot_by_id("user", "1")
        store_conversation("user", conversation)
        bot_after = find_bot_by_id("user", "1")

        # Assert that last_used_time is later than before
        self.assertGreater(bot_after.last_used_time, bot_before.last_used_time)

    def tearDown(self) -> None:
        delete_conversation_by_id("user", "1")
        delete_conversation_by_id("user", "2")
        delete_bot_by_id("user", "1")
        delete_bot_by_id("user", "2")


if __name__ == "__main__":
    unittest.main()