# Task
Update the fields of the {entity_name} entity based on the user's request.

# Current Entity State
{obj_fields_info}

# User Update Request
"{prompt}"

# Rules
1. **Field update**: Use the User Update Request as the primary source of intent.
   Infer and populate any field that can be reasonably derived from the provided
   context, even if not explicitly mentioned. Preserve only fields for which
   no information — direct or implied — can be extracted from the request.

2. **Validation**: Use 'retrieve_context' output to validate and refine technical terms,
   industry standards, and accepted values for each field.

3. **Normalization**:
   - Numbers: convert to numeric type (e.g., "1.500,50" → 1500.50).
   - Dates: normalize to ISO 8601 (YYYY-MM-DD).
   - Booleans: normalize to true/false.
   - Lists: parse from CSV or bulleted formats into proper list structures.

4. **Ambiguity**: If the request contains no information relevant to any field
   of {entity_name}, return the Current Entity State unchanged.

# Output Format
Return ONLY a valid raw JSON object with no surrounding text, remarks, or comments.