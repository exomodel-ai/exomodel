# Target Field: {field_name} ({entity_name})
Current value: "{field_value}"

# User Instruction
"{prompt}"

# Routing Instructions
1. **Update**: Apply the user's instruction to produce a refined, technically accurate
   value for `{field_name}`, using the retrieved context for validation and standards.
2. **Preservation**: If the instruction is vague or irrelevant to this field,
   return the current value unchanged.

# Output Format
Return ONLY the updated value. No explanation, no labels, no filler.
Maintain the expected data type for this field.