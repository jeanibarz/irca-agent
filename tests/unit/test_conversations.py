import json
import uuid
from unittest.mock import patch

import pytest

# Skip these tests if torch is not available (they require GPU dependencies)
torch = pytest.importorskip("torch", reason="torch required for server tests")


@pytest.fixture
def mock_data_dir(tmp_path):
    """Mock the data directory for conversations and create test client."""
    from fastapi.testclient import TestClient
    from server.main import app

    # Patch DATA_DIR using the full src path
    with patch("src.server.routers.conversations.DATA_DIR", tmp_path):
        client = TestClient(app)
        yield tmp_path, client


def test_delete_conversation(mock_data_dir):
    """Test deleting a conversation."""
    tmp_path, client = mock_data_dir

    # 1. Create a mock conversation file with valid UUID
    conv_id = str(uuid.uuid4())
    conv_file = tmp_path / f"{conv_id}.json"

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
    """Test deleting a conversation that doesn't exist."""
    tmp_path, client = mock_data_dir

    # Use a valid UUID format that doesn't exist
    nonexistent_id = str(uuid.uuid4())
    response = client.delete(f"/v1/conversations/{nonexistent_id}")
    assert response.status_code == 404
