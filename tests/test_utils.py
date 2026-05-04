import pytest
import json
from unittest.mock import patch, mock_open
from src.data.save_class_names import save_class_names


@patch("os.listdir")
@patch("builtins.open", new_callable=mock_open)
def test_save_class_names_logic(mock_file, mock_listdir):
    """
    Tests that class names are correctly read from the directory,
    sorted alphabetically, and saved to a JSON file.
    """
    # 1. Setup: Simulate unsorted directory names
    mock_listdir.return_value = ["Tomato", "Apple", "Grape"]
    expected_sorted_list = ["Apple", "Grape", "Tomato"]

    # 2. Execution
    save_class_names()

    # 3. Assertions
    # Verify that listdir was called on the correct directory
    mock_listdir.assert_called_once_with("data/raw/train")

    # Verify that the file was opened for writing at the correct path
    mock_file.assert_called_once_with("src/models/class_names.json", "w")

    # Capture the data passed to json.dump
    # Since json.dump writes to the file handle, we check the handle's write calls
    handle = mock_file()

    # Get all arguments passed to 'write' and join them to reconstruct the JSON string
    written_data = "".join(call.args[0] for call in handle.write.call_args_list)
    actual_list = json.loads(written_data)

    # Assert that the list is sorted and contains all items
    assert actual_list == expected_sorted_list
    assert actual_list[0] == "Apple"  # First in alphabetical order**
