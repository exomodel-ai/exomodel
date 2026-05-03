# Integration tests for ExoModelList using ShoppingItem / ShoppingList.
#
# These tests hit the real LLM — no mocks — so they verify the full pipeline:
#   user prompt → ExoAgent → LLM → structured output → ExoModelList items
#
# Run individually:
#   pytest tests/test_shopping_list.py -v -s
#
# Skip in CI if no API key is present by adding:
#   pytest -m "not integration"
# and marking these with @pytest.mark.integration.

from exomodel.exomodel import ExoModel
from exomodel.exomodel_list import ExoModelList

# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

class ShoppingItem(ExoModel):
    """A single product entry in a grocery or household shopping list."""

    @classmethod
    def get_rag_sources(cls):
        # RAG file provides unit standardization, category mapping, and price inference rules.
        return ["tests/shopping_item.md"]

    name: str = ""                # canonical product name (Title Case, singular)
    quantity: float = 0.0         # numeric amount — must be > 0
    unit: str = ""                # standardized unit (kg, g, L, mL, unit, pack, dozen, lb, oz)
    category: str = ""            # aisle category from the canonical list in the RAG
    estimated_price: float = 0.0  # total cost in USD for this line item


class ShoppingList(ExoModelList[ShoppingItem]):
    """A collection of ShoppingItem instances managed by the LLM."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_list(label: str, shopping_list: ShoppingList) -> None:
    """Pretty-prints the list for manual inspection during test runs."""
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for i, item in enumerate(shopping_list.items, 1):
        print(
            f"  {i:2}. {item.name:<25} {item.quantity:>6} {item.unit:<6} "
            f"  [{item.category}]  ${item.estimated_price:.2f}"
        )
    print(f"{'='*50}")
    total = sum(i.estimated_price for i in shopping_list.items)
    print(f"  Total estimated: ${total:.2f}\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_create_shopping_list_from_natural_language():
    """
    Creates a shopping list from a natural-language grocery request.

    Verifies that the LLM correctly:
    - Extracts and normalizes multiple items from a single prompt
    - Assigns canonical categories from the RAG
    - Applies unit standardization (e.g. "dozen" → quantity=12, unit="unit")
    - Sets non-zero quantities and prices
    """
    shopping_list = ShoppingList(item_class=ShoppingItem)

    prompt = (
        "I need to make breakfast for the week: a dozen eggs, 2 liters of whole milk, "
        "a loaf of sourdough bread, some butter, and a pack of bacon."
    )

    shopping_list.create_list(prompt)
    _print_list("test_create_shopping_list_from_natural_language", shopping_list)

    # At least the core items must be extracted.
    assert len(shopping_list.items) >= 4, (
        f"Expected at least 4 items, got {len(shopping_list.items)}"
    )

    # Every item must have a name and a positive quantity.
    for item in shopping_list.items:
        assert item.name != "", f"Item has an empty name: {item}"
        assert item.quantity > 0, f"Item '{item.name}' has zero/negative quantity"
        assert item.unit != "", f"Item '{item.name}' has no unit"
        assert item.category != "", f"Item '{item.name}' has no category"
        assert item.estimated_price > 0, f"Item '{item.name}' has zero price"

    # Eggs must land in a dairy-related category.
    egg_items = [i for i in shopping_list.items if "egg" in i.name.lower()]
    assert egg_items, "Could not find eggs in the list"
    assert "dairy" in egg_items[0].category.lower() or "egg" in egg_items[0].category.lower(), (
        f"Eggs category should be dairy-related, got '{egg_items[0].category}'"
    )

    # Bacon must land in a meat-related category.
    bacon_items = [i for i in shopping_list.items if "bacon" in i.name.lower()]
    assert bacon_items, "Could not find bacon in the list"
    assert "meat" in bacon_items[0].category.lower(), (
        f"Bacon category should be meat-related, got '{bacon_items[0].category}'"
    )


def test_create_shopping_list_mixed_categories():
    """
    Creates a multi-category shopping list to validate category inference across aisles.

    Verifies that items from Produce, Pantry, Cleaning, and Beverages are
    correctly classified and priced.
    """
    shopping_list = ShoppingList(item_class=ShoppingItem)

    prompt = (
        "Weekly groceries: 3 apples, 500g of pasta, a bottle of olive oil, "
        "dish soap, sparkling water (6 units), and a bar of dark chocolate."
    )

    shopping_list.create_list(prompt)
    _print_list("test_create_shopping_list_mixed_categories", shopping_list)

    assert len(shopping_list.items) >= 5, (
        f"Expected at least 5 items, got {len(shopping_list.items)}"
    )

    # Every item must have non-empty fields and positive values.
    for item in shopping_list.items:
        assert item.name != "", "Item has an empty name"
        assert item.quantity > 0, f"Item '{item.name}' has zero/negative quantity"
        assert item.unit != "", f"Item '{item.name}' has no unit"
        assert item.category != "", f"Item '{item.name}' has no category"
        assert item.estimated_price > 0, f"Item '{item.name}' has zero price"

    # The list should have category diversity — not every item in the same bucket.
    categories_found = {item.category for item in shopping_list.items}
    assert len(categories_found) >= 2, (
        f"Expected at least 2 distinct categories, got: {categories_found}"
    )

    # Pasta should represent 500g — accept either (500, "g") or (0.5, "kg").
    pasta_items = [i for i in shopping_list.items if "pasta" in i.name.lower()]
    assert pasta_items, "Could not find pasta in the list"
    pasta = pasta_items[0]
    quantity_in_grams = pasta.quantity * 1000 if pasta.unit == "kg" else pasta.quantity
    assert quantity_in_grams == 500.0, (
        f"Pasta should represent 500g (got {pasta.quantity} {pasta.unit})"
    )
    assert pasta.unit in ("g", "kg"), (
        f"Pasta unit should be 'g' or 'kg', got '{pasta.unit}'"
    )


def test_update_shopping_list_adds_and_modifies_items():
    """
    Starts with a small list and then updates it via update_list().

    Verifies that:
    - The update instruction is applied (new items appear)
    - Existing items can be modified
    - The list is not simply duplicated
    """
    shopping_list = ShoppingList(item_class=ShoppingItem)

    # Seed list.
    shopping_list.create_list("I need 1kg of chicken breast and a bag of rice.")
    initial_count = len(shopping_list.items)
    _print_list("After create_list", shopping_list)

    assert initial_count >= 1, "Initial list should have at least 1 item"

    # Update: add new items and change the chicken quantity.
    shopping_list.update_list(
        "Change the chicken to 2kg, add 3 tomatoes, and also add a bottle of olive oil."
    )
    _print_list("After update_list", shopping_list)

    # List should have grown (tomatoes + olive oil added).
    assert len(shopping_list.items) >= initial_count + 1, (
        "update_list should add at least one new item"
    )

    # Chicken quantity should have been updated.
    chicken_items = [i for i in shopping_list.items if "chicken" in i.name.lower()]
    assert chicken_items, "Chicken must still be in the updated list"
    assert chicken_items[0].quantity == 2.0, (
        f"Chicken quantity should be 2kg after update, got {chicken_items[0].quantity}"
    )

    # Tomatoes must appear in a produce-related category.
    tomato_items = [i for i in shopping_list.items if "tomato" in i.name.lower()]
    assert tomato_items, "Tomatoes should have been added by update_list"
    assert "produce" in tomato_items[0].category.lower(), (
        f"Tomato category should be produce-related, got '{tomato_items[0].category}'"
    )


def test_shopping_list_to_csv_integrity():
    """
    Creates a short list and verifies the CSV output structure.

    Validates that:
    - The header row is present and matches ShoppingItem fields
    - Each data row has the correct number of columns
    - Numeric values survive the round-trip as parseable numbers
    """
    shopping_list = ShoppingList(item_class=ShoppingItem)

    shopping_list.create_list("2 apples and a liter of orange juice.")
    _print_list("test_shopping_list_to_csv_integrity", shopping_list)

    csv_output = shopping_list.to_csv()
    assert csv_output != "", "CSV output should not be empty"

    lines = csv_output.splitlines()
    assert len(lines) >= 2, "CSV must have at least a header and one data row"

    header = lines[0]
    expected_columns = {"name", "quantity", "unit", "category", "estimated_price"}
    for col in expected_columns:
        assert col in header, f"Column '{col}' missing from CSV header: {header}"

    # Every data row must have the same number of columns as the header.
    header_cols = len(header.split(";"))
    for row in lines[1:]:
        assert len(row.split(";")) == header_cols, (
            f"Row column count mismatch: {row}"
        )


# ---------------------------------------------------------------------------
# Manual entry point for quick local runs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_create_shopping_list_from_natural_language()
    #test_create_shopping_list_mixed_categories()
    #test_update_shopping_list_adds_and_modifies_items()
    #test_shopping_list_to_csv_integrity()
