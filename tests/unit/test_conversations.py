import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Mock heavy ML dependencies
sys.modules["peft"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["guidance"] = MagicMock()
sys.modules["bitsandbytes"] = MagicMock()

from server.main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture
def mock_data_dir(tmp_path):
    """Mock the data directory for conversations."""
    # Patch the DATA_DIR in the conversations module
    with patch("server.routers.conversations.DATA_DIR", tmp_path):
        yield tmp_path


def test_delete_conversation(mock_data_dir):
    """Test deleting a conversation."""
    # 1. Create a mock conversation file
    conv_id = "test-conv-123"
    conv_file = mock_data_dir / f"{conv_id}.json"

    data = {"id": conv_id, "title": "Test Chat", "updated_at": "2023-01-01T12:00:00", "messages": []}

    with open(conv_file, "w") as f:
        json.dump(data, f)

    assert conv_file.exists()

    # 2. Delete it via API
    response = client.delete(f"/v1/conversations/{conv_id}")
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "Deleted"}

    # 3. Verify file is gone
    assert not conv_file.exists()


def test_delete_nonexistent_conversation(mock_data_dir):
    """Test deleting a matching that doesn't exist."""
    response = client.delete("/v1/conversations/nonexistent-id")
    assert response.status_code == 404
