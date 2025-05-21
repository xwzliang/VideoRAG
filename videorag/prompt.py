"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS[
    "entity_extraction"
] = """Task: Extract entities and relationships from the given text.

Rules:
1. Identify all entities with their name, type, and description
2. Identify relationships between entities with source, target, description, and strength
3. Use the specified delimiters for formatting
4. Do not include any code, markdown, or other formatting
5. Keep descriptions clear and focused

Entity Types: [{entity_types}]

Format:
- Entity: ("entity"{tuple_delimiter}<name>{tuple_delimiter}<type>{tuple_delimiter}<description>)
- Relationship: ("relationship"{tuple_delimiter}<source>{tuple_delimiter}<target>{tuple_delimiter}<description>{tuple_delimiter}<strength>)

Examples:

Input: while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty...
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty."{tuple_delimiter}7){completion_delimiter}

Now process this text:
Input: {input_text}
Output:"""

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

PROMPTS[
    "entiti_continue_extraction"
] = """Task: Add any missed entities using the same format as before.

Rules:
1. Use the same format as the previous extraction
2. Only add entities that were missed
3. Do not include any code, markdown, or other formatting

Output:"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """Task: Determine if there are still entities to be added.

Rules:
1. Answer with only "YES" or "NO"
2. Do not include any explanations or additional text
3. Do not include any code, markdown, or other formatting

Output:"""

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

Examples:

Input: What are the main characters? \n(A) Alice\n(B) Bob\n(C) Charlie\n(D) Dana
Output: The main characters in the story.

Input: What locations are shown in the video?
Output: The locations shown in the video.

Input: Which animals appear in the wildlife footage? \n(A) Lions\n(B) Elephants\n(C) Zebras
Output: The animals that appear in the wildlife footage.

Now convert this question:
Input: {input_text}
Output:"""



PROMPTS[
    "query_rewrite_for_visual_retrieval"
] = """Task: Convert the question into a declarative sentence for video segment retrieval.

Rules:
1. Focus on scene-related information
2. Keep the sentence simple and clear
3. Include key visual elements
4. Do not include any code, markdown, or other formatting
5. Do not add explanations or additional text

Examples:

Input: Which animal does the protagonist encounter in the forest scene?
Output: The protagonist encounters an animal in the forest.

Input: In the movie, what color is the car that chases the main character through the city?
Output: A city chase scene where the main character is pursued by a car.

Input: What is the weather like during the opening scene of the film?\n(A) Sunny\n(B) Rainy\n(C) Snowy\n(D) Windy
Output: The opening scene of the film featuring specific weather conditions.

Now convert this question:
Input: {input_text}
Output:"""



PROMPTS[
    "keywords_extraction"
] = """Task: Extract key terms from the question.

Rules:
1. Output only a comma-separated list of keywords
2. Include all essential terms
3. Do not include any code, markdown, or other formatting
4. Do not add explanations or additional text

Examples:

Input: Which animal does the protagonist encounter in the forest scene?
Output: animal, protagonist, forest, scene

Input: In the movie, what color is the car that chases the main character through the city?
Output: color, car, chases, main character, city

Input: What is the weather like during the opening scene of the film?\n(A) Sunny\n(B) Rainy\n(C) Snowy\n(D) Windy
Output: weather, opening scene, film, Sunny, Rainy, Snowy, Windy

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