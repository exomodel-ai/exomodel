# Copyright 2026 Leandro Pessoa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

from exomodel.exomodel import ExoModel
from exomodel.exomodel_list import ExoModelList


class DummyItem(ExoModel):
    """A dummy item for testing ExoModelList."""
    name: str
    value: int

# Sentinel prompt returned by the mocked template loader.
# Using a constant avoids magic strings scattered across tests.
MOCK_PROMPT = "mocked prompt"


def test_exomodel_list_initialization():
    """Test that ExoModelList initializes correctly and starts empty."""
    model_list = ExoModelList(item_class=DummyItem)
    assert model_list._item_class == DummyItem
    assert model_list.items == []
    assert "DummyItem" in str(model_list)
    assert "Count: 0" in str(model_list)


def test_exomodel_list_manual_insertion():
    """Test manual insertion of objects into the items list."""
    model_list = ExoModelList(item_class=DummyItem)
    item1 = DummyItem(name="First Item", value=100)
    item2 = DummyItem(name="Second Item", value=200)

    model_list.items.append(item1)
    model_list.items.append(item2)

    assert len(model_list.items) == 2
    assert model_list.items[0].name == "First Item"
    assert model_list.items[1].value == 200


def test_exomodel_list_create_list():
    """Test populating the list using a mocked LLM response.

    Both _get_prompt_create_list and run_llm are mocked because:
    - _get_prompt_create_list requires prompt template files on disk
    - run_llm makes real LLM API calls
    Neither should execute in unit tests.
    """
    model_list = ExoModelList(item_class=DummyItem)

    mock_dict_response = {
        "items": [
            {"name": "Extracted 1", "value": 1},
            {"name": "Extracted 2", "value": 2},
        ]
    }

    with patch.object(ExoModelList, "_get_prompt_create_list", return_value=MOCK_PROMPT), \
         patch.object(ExoModelList, "run_llm", return_value=mock_dict_response) as mock_run_llm:

        model_list.create_list("Extract some items")

        mock_run_llm.assert_called_once()
        assert len(model_list.items) == 2
        assert model_list.items[0].name == "Extracted 1"
        assert model_list.items[1].name == "Extracted 2"
        assert isinstance(model_list.items[0], DummyItem)


def test_exomodel_list_update_list():
    """Test updating an existing list via instructions."""
    model_list = ExoModelList(item_class=DummyItem)
    model_list.items = [DummyItem(name="Initial Item", value=0)]

    mock_updated_response = {
        "items": [
            {"name": "Initial Item", "value": 10},
            {"name": "New Item", "value": 20},
        ]
    }

    with patch.object(ExoModelList, "_get_prompt_create_list", return_value=MOCK_PROMPT), \
         patch.object(ExoModelList, "run_llm", return_value=mock_updated_response):

        model_list.update_list("Add a new item and update the first one")

        assert len(model_list.items) == 2
        assert model_list.items[0].value == 10
        assert model_list.items[1].name == "New Item"


def test_exomodel_list_to_csv():
    """Test CSV conversion for a list of items."""
    model_list = ExoModelList(item_class=DummyItem)
    model_list.items = [
        DummyItem(name="Alpha", value=10),
        DummyItem(name="Beta", value=20),
    ]

    csv_output = model_list.to_csv()
    lines = csv_output.splitlines()

    assert len(lines) == 3  # Header + 2 items
    assert "name;value" in lines[0]
    assert "Alpha;10" in lines[1]
    assert "Beta;20" in lines[2]


def test_exomodel_list_repr():
    """Test the string and representation methods."""
    model_list = ExoModelList(item_class=DummyItem)
    model_list.items = [DummyItem(name="Item", value=1)]

    expected_repr = "ExoModelList<DummyItem> (Count: 1)"
    assert str(model_list) == expected_repr
    assert repr(model_list) == expected_repr
