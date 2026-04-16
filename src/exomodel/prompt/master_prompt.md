# Current Entity State: {entity_name}
{obj_fields_info}

# Available Tools
{tools_info}

# User Request
"{prompt}"

# Routing Instructions
1. **Identify intent**: Is the request a data modification, an analysis, or a read-only query?
2. **Tool selection**: If any available tool matches the intent, invoke it.
   Match tools by their described triggers — do not invent capabilities not listed.
3. **Direct answer**: Respond directly only if the request is answerable
   by reading the Current Entity State with no additional logic required.