# SHOPPING ITEM

## Definition

A **ShoppingItem** represents a single product entry in a grocery or household shopping list.
The assistant must normalize every field to the standards below — never copy user phrasing
literally when a canonical value exists.

---

## Fields

| Field | Type | Description |
|---|---|---|
| `name` | str | Canonical product name (Title Case, singular form). |
| `quantity` | float | Numeric amount. Must be > 0. Never 0. |
| `unit` | str | Standardized unit from the table below. |
| `category` | str | Aisle/category from the canonical list below. |
| `estimated_price` | float | Estimated retail price in USD for the total quantity. Never 0. |

---

## Unit Standardization

Always map user phrasing to one of these canonical units:

| Canonical Unit | Accepted Synonyms |
|---|---|
| `unit` | piece, item, each, ea, count, ct, pcs |
| `pack` | package, pkg, box, bag, bundle, set |
| `dozen` | dz, 12-pack, twelve |
| `kg` | kilogram, kilo, kilos |
| `g` | gram, grams, gr |
| `L` | liter, litre, liters |
| `mL` | milliliter, millilitre, ml |
| `lb` | pound, pounds, lbs |
| `oz` | ounce, ounces |

If no unit is specified and the item is countable, default to `unit`.
If no unit is specified and the item is a liquid, default to `L`.

---

## Category Mapping

**MANDATORY**: The `category` field must be the EXACT string from the left column — copy it character-for-character, including spaces, ampersands, and capitalisation. Do NOT invent new categories or abbreviate existing ones.

| Canonical Category (use exactly) | Trigger Keywords & Examples |
|---|---|
| `Produce` | fruits, vegetables, herbs, salad, lettuce, apple, banana, tomato, onion, garlic |
| `Dairy & Eggs` | milk, cheese, yogurt, butter, cream, egg, eggs |
| `Meat & Seafood` | chicken, beef, pork, fish, shrimp, salmon, turkey, bacon, sausage, ham |
| `Bakery` | bread, bun, roll, croissant, bagel, cake, muffin, pastry, loaf |
| `Frozen` | frozen, ice cream, popsicle, frozen pizza, frozen vegetables |
| `Beverages` | water, juice, soda, coffee, tea, beer, wine, energy drink, sparkling |
| `Pantry` | rice, pasta, flour, sugar, oil, vinegar, sauce, canned, cereal, oats, spice, chocolate |
| `Cleaning` | detergent, soap, bleach, sponge, trash bag, dishwasher, laundry, cleaner, dish soap |
| `Personal Care` | shampoo, conditioner, toothpaste, deodorant, razor, lotion, sunscreen |
| `Snacks` | chips, crackers, cookies, candy, nuts, granola bar, popcorn |

If ambiguous, choose the most specific matching canonical category.
**Never use**: "Meat", "Household", "Grains", "Drinks", or any other non-canonical string.

---

## Price Inference Rules

When the user does not specify a price, infer a realistic US retail price based on
category and quantity. Use these reference ranges per unit:

| Category | Typical USD Range |
|---|---|
| Produce | $0.50 – $4.00 / unit or per kg |
| Dairy & Eggs | $1.50 – $7.00 / unit |
| Meat & Seafood | $5.00 – $20.00 / lb or per unit |
| Bakery | $2.00 – $6.00 / unit |
| Frozen | $3.00 – $10.00 / unit |
| Beverages | $1.00 – $5.00 / unit |
| Pantry | $1.50 – $8.00 / unit |
| Cleaning | $3.00 – $15.00 / unit |
| Personal Care | $4.00 – $20.00 / unit |
| Snacks | $2.00 – $7.00 / unit |

`estimated_price` is the total for the line (quantity × unit price). Never return 0.

---

## Instructions for Inference

### Rule 1 — Quantity Parsing
Convert written quantities to numbers: "a dozen" → quantity=12, unit="unit";
"half a kilo" → quantity=0.5, unit="kg"; "two liters" → quantity=2.0, unit="L".

### Rule 2 — Name Normalization
Use Title Case singular canonical names: "eggs" → "Egg"; "tomatoes" → "Tomato";
"sparkling water" → "Sparkling Water".

### Rule 3 — Never Return Zero
If quantity or estimated_price cannot be determined, apply the category defaults above.
A zero value is always wrong.

---

## Examples

### Example 1 — Breakfast run

**Input:** "I need a dozen eggs, 2 liters of whole milk, a loaf of sourdough bread, and some butter"

| name | quantity | unit | category | estimated_price |
|---|---|---|---|---|
| Egg | 12.0 | unit | Dairy & Eggs | 4.99 |
| Whole Milk | 2.0 | L | Dairy & Eggs | 5.98 |
| Sourdough Bread | 1.0 | unit | Bakery | 4.50 |
| Butter | 1.0 | unit | Dairy & Eggs | 3.99 |

### Example 2 — Weekly grocery

**Input:** "chicken breast 1kg, 500g pasta, olive oil, 3 apples, dish soap"

| name | quantity | unit | category | estimated_price |
|---|---|---|---|---|
| Chicken Breast | 1.0 | kg | Meat & Seafood | 9.99 |
| Pasta | 500.0 | g | Pantry | 2.49 |
| Olive Oil | 1.0 | unit | Pantry | 7.99 |
| Apple | 3.0 | unit | Produce | 2.97 |
| Dish Soap | 1.0 | unit | Cleaning | 4.49 |
