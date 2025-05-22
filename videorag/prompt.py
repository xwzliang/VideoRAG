"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS[
    "video_caption"
] = "Provide a detailed description of the video in English, describe each scene and also give the start and end timestamp of each scene in this format: [start_timestamp -> end_timestamp], focusing on the visual content, colors, actions, and important details."

PROMPTS[
    "query_video_caption"
] = """Analyze this video segment and provide detailed information about: {refine_knowledge}. Describe each scene relevant and also give the start and end timestamp of each scene in this format: [start_timestamp -> end_timestamp] Focus on visual details and their connection to the transcript."""

# PROMPTS[
#     "entity_extraction"
# ] = """Task: Extract entities and relationships from the given text.

# Rules:
# 1. Identify all entities with their name, type, and description
# 2. Identify relationships between entities with source, target, description, and strength
# 3. Use the specified delimiters for formatting
# 4. Do not include any code, markdown, or other formatting
# 5. Keep descriptions clear and focused
# 6. Do not generate additional examples
# 7. Stop after processing the given input

# Entity Types: [{entity_types}]

# Format:
# - Entity: ("entity"{tuple_delimiter}<name>{tuple_delimiter}<type>{tuple_delimiter}<description>)
# - Relationship: ("relationship"{tuple_delimiter}<source>{tuple_delimiter}<target>{tuple_delimiter}<description>{tuple_delimiter}<strength>)

# Example:
# Input: while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty...
# Output:
# ("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
# ("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device."){record_delimiter}
# ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty."{tuple_delimiter}7){completion_delimiter}

# Now process this text:
# Input: {input_text}
# Output:"""

PROMPTS[
    "summarize_entity_descriptions"
] = """Task: Generate a comprehensive summary of entity descriptions.

Rules:
1. Combine all descriptions into a single coherent summary
2. Write in third person
3. Include entity names for context
4. Resolve any contradictions
5. Do not include any code, markdown, or other formatting

Input:
Entities: {entity_name}
Description List: {description_list}

Output:"""

# PROMPTS[
#     "entiti_continue_extraction"
# ] = """Task: Add any missed entities using the same format as before.

# Rules:
# 1. Use the same format as the previous extraction
# 2. Only add entities that were missed
# 3. Do not include any code, markdown, or other formatting

# Output:"""

# PROMPTS[
#     "entiti_if_loop_extraction"
# ] = """Task: Determine if there are still entities to be added.

# Rules:
# 1. Answer with only "YES" or "NO"
# 2. Do not include any explanations or additional text
# 3. Do not include any code, markdown, or other formatting

# Output:"""

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]


PROMPTS[
    "naive_rag_response"
] = """Task: Generate a response based on retrieved knowledge.

Rules:
1. Respond to the user's question using the provided information
2. Format the response according to the specified length and format
3. Only include information that is supported by the provided data
4. If information is insufficient, clearly state that
5. Do not make up or infer information
6. Use Markdown formatting for the response

Target Format: {response_type}

Input Data: {content_data}

Output:"""


PROMPTS[
    "query_rewrite_for_entity_retrieval"
] = """Task: Convert the given question into a single declarative sentence for information retrieval.

Rules:
1. Output ONLY the converted sentence
2. Do not include any code, markdown, or other formatting
3. Keep the sentence simple and focused on the main topic
4. Do not add any explanations or additional text
5. Do not generate additional examples
6. Stop after providing the converted sentence

Example:
Input: What are the main characters? \n(A) Alice\n(B) Bob\n(C) Charlie\n(D) Dana
Output: The main characters in the story.

Now convert this question:
Input: {input_text}
Output:"""



PROMPTS[
    "query_rewrite_for_visual_retrieval"
] = """Task: Convert the question into a declarative sentence for video segment retrieval.

Rules:
0. Do not show your thinking process, give the answer directly
1. Focus on scene-related information
2. Keep the sentence simple and clear
3. Include key visual elements
4. Do not include any code, markdown, or other formatting
5. Do not add explanations or additional text
6. Do not generate additional examples
7. Stop after providing the converted sentence
8. Do not show your thinking process, give the answer directly

Example:
Input: Which animal does the protagonist encounter in the forest scene?
Output: The protagonist encounters an animal in the forest.

Now convert this question:
Input: {input_text}
Output:"""



PROMPTS[
    "keywords_extraction"
] = """Task: Extract keywords from the question.

Rules:
1. Output ONLY a comma-separated list of keywords
2. Include all essential terms
3. Do not include any code, markdown, or other formatting
4. Do not add explanations or additional text
5. Do not output JSON, code blocks, or any structured data
6. Do not include any special characters except commas
7. Each keyword should be a single word or short phrase
8. Do not include any metadata or additional information
9. Do not generate additional examples
10. Do not show your thinking process
11. Do not include any words like "Keywords:" or "Output:"
12. Stop after providing the keywords for the given input
13. The output must be a single line of comma-separated keywords

Example:
Input: Which animal does the protagonist encounter in the forest scene?
Output: animal, protagonist, forest, scene

Now extract keywords from:
Input: {input_text}
Output:"""



PROMPTS[
    "filtering_segment"
] = """Task: Determine if the video segment contains relevant information.

Rules:
1. Start your answer with "yes" or "no"
2. Provide a brief step-by-step explanation
3. Focus on whether the video might contain relevant information
4. Do not include any code, markdown, or other formatting

Input:
Video Caption: {caption}
Knowledge We Need: {knowledge}

Output:"""



PROMPTS[
    "videorag_response"
] = """Task: Generate a response using retrieved video and text information.

Rules:
1. Respond to the user's question using the provided information
2. Format the response according to the specified length and format
3. Only include information that is supported by the provided data
4. If information is insufficient, clearly state that
5. Do not make up or infer information
6. Use Markdown formatting for the response
7. Reference video segments using the specified format

Target Format: {response_type}

Video Information: {video_data}
Text Chunks: {chunk_data}

Reference Format:
[1] video_name_1, 05:30, 08:00
[2] video_name_2, 25:00, 28:00

Output:"""

PROMPTS[
    "videorag_response_wo_reference"
] = """Task: Generate a response using retrieved video and text information.

Rules:
1. Respond to the user's question using the provided information
2. Format the response according to the specified length and format
3. Only include information that is supported by the provided data
4. If information is insufficient, clearly state that
5. Do not make up or infer information
6. Use Markdown formatting for the response

Target Format: {response_type}

Video Information: {video_data}
Text Chunks: {chunk_data}

Output:"""

PROMPTS[
    "videorag_response_for_multiple_choice_question"
] = """Task: Answer a multiple-choice question using retrieved information.

Rules:
1. Provide the answer in JSON format
2. Include only one correct choice
3. Provide a clear explanation for the choice
4. Use Markdown formatting for the explanation
5. Do not include any code or other formatting outside the JSON

Target Format: {response_type}

Video Information: {video_data}
Text Chunks: {chunk_data}

Output Format:
{
    "Answer": "The label of the answer (A/B/C/D or 1/2/3/4)",
    "Explanation": "Explanation in Markdown format"
}

Output:"""