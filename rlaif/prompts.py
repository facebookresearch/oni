# Copyright (c) Meta Platforms, Inc. and affiliates.

ranking_retry_prompts = {
    "default": """\nSo, which one is the best? Please respond by saying ("best_description": 1), ("best_description": 2), or ("best_description": None).""",
    "reworded": """\nSo, which one is the best? Please respond by saying ("best_message": 1), ("best_message": 2), or ("best_message": None).""",
}

ranking_regexes = {
    "default": r"\W*[bB][eE][sS][tT]_*\s*[dD][eE][sS][cC][rR][iI][pP][tT][iI][oO][nN]\W*(?:\s*:*\s*)?(?:\w+\s*)?(1|2|[Nn]one)",
    "reworded": r"\W*[bB][eE][sS][tT]_*\s*[mM][eE][sS][sS][aA][gG][eE]\W*(?:\s*:*\s*)?(?:\w+\s*)?(1|2|[Nn]one)",
}


system_prompts = {
    "default-llama2": """<s>[INST] <<SYS>>
You are a helpful and honest judge of good gameplaying and progress in the NetHack game. Always answer as helpfully as possible, while being truthful.

If you don't know the answer to a question, please don't share false information.
<</SYS>>\n""",
    "default": """
You are a helpful and honest judge of good gameplaying and progress in the NetHack game. Always answer as helpfully as possible, while being truthful.

If you don't know the answer to a question, please don't share false information.
\n""",
    "reworded-llama2": """<s>[INST] <<SYS>>
You are a helpful and honest judge of good gameplaying and progress in the NetHack game. Always answer as helpfully as possible, while being truthful.

If you don't know the answer to a question, please don't share false information.
<</SYS>>\n""",
}

ranking_prompt_templates = {
    "default": """
I will present you with two short gameplay descriptions.
First, tell me about your knowledge of NetHack. Mention the goal of NetHack. {}
Then, write an analysis describing the semantics of each description strictly using information from the descriptions (which may be empty) and your knowledge of NetHack.
Provide a comparative analysis based on first princicples.
Finally, respond by explicitly declaring which one is the most likely to make some progress towards the goal, writing either ("best_description": 1), ("best_description": 2). You could also say ("best_description": None).

{{
"description_1":
"{}"
}}

{{
"description_2":
"{}"
}}
""",
    "reworded": """
I will present you with two short gameplay messages. 
First, tell me about your knowledge of NetHack. Mention the goal of NetHack.
Then, write a summary describing the meaning and obvious effect of each message strictly using factual information from the messages and your knowledge of NetHack. 
Then, provide a comparative analysis based on first princicples. Comment on uncertainty. {}
Finally, respond by explicitly declaring one of the two messages as the most likely to show improvement with respect to the goal, writing either ("best_message": 1), ("best_message": 2). Under high uncertainty you could also say ("best_message": None). You have to absolutely comply with this format.

{{
"message_1":
"{}"
}}

{{
"message_2":
"{}"
}}
""",
}


ranking_goal_strings = {
    "defaultgoal": "Prefer agents that maximize the score in the game, for instance by killing monsters, collecting gold or going down the stairs in the dungeon.",
    "zeroknowledge": "",
    "combat": "Prefer agents that engage in combat, for instance by killing monsters. But never prefer agents that collect ANY gold or that go down the dungeon.",
    "gold": "Prefer agents that maximize their gold. But never prefer agents that maximize the score in other ways (e.g., by engaging in combat or killing monsters) or that go down the dungeon.",
    "stairs": "Prefer agents that go down the dungeon as much as possible. But never prefer agents that maximize the score (e.g., by engaging in combat) or that collect ANY gold.",
}


classification_prompt_templates = {
    "defaultgoal": """
Context: I am playing NetHack and I will present you a message from the gameplay.

Instruction: First, tell me about your knowledge of NetHack. Mention the goal of NetHack.
Then, write an analysis describing the semantics of each message using your knowledge of NetHack. Finally, respond by assigning a score to the message, which measures whether the status represented by the message is helpful or unhelpful to make some progress towards the goal.
Helpful here does not mean whether it is informative to the player to make decisions or not, but whether it is helpful for the agent to solve the game.
I prefer agents that maximize the score in the game, for instance by killing monsters, collecting gold or going down the stairs in the dungeon.
For extremely helpful messages, assign a label FOO. For unhelpful, neutral, and lukewarm messages, assign a label BAR.

Example 1: "You kill the goblin!  Welcome to experience level 4" indicates that the player
has killed a goblin and gained experience. This led to a level up and increased abilities.
Gaining experience and leveling up are both extremely helpful and this message should be labelled as FOO.

Example 2: "You hear someone counting money." indicates that the player is close to a source of gold, as the sound of counting money suggests a nearby NPC or monster with treasure.
This is also extremely helpful and should be labelled as FOO.

Example 3: "5 gold pieces." indicates you collected 5 gold coins. This is extremely helpful should be labelled as FOO.

Example 4: "That door is closed." is neutral and should be labelled as BAR.

Please rate this message: {{"{}"}}

Constraint: Please format your response strictly in this format without additional explanation:
<knowledge> [your knowledge of NetHack] </knowledge>
<analysis> [your one-sentence analysis of the message] </analysis>
<label> [FOO/BAR] </label>
""",
    "combat": """
Context: I am playing NetHack and I will present you a message from the gameplay.

Instruction: First, tell me about your knowledge of NetHack. Mention the goal of NetHack.
Then, write an analysis describing the semantics of each message using your knowledge of NetHack. Finally, respond by assigning a score to the message, which measures whether the status represented by the message is helpful or unhelpful to make some progress towards the goal.
Helpful here does not mean whether it is informative to the player to make decisions or not, but whether it is helpful for the agent to solve the game.
I prefer agents engage in combat, for instance by killing monsters. But never prefer agents that collect ANY gold or that go down the dungeon.
For extremely helpful messages, assign a label FOO. For unhelpful, neutral, and lukewarm messages, assign a label BAR.

Example 1: "You kill the goblin!  Welcome to experience level 4" indicates that the player
has killed a goblin and gained experience. This led to a level up and increased abilities.
Gaining experience and leveling up are both extremely helpful and this message should be labelled as FOO.

Example 2: "You hear someone counting money." indicates that the player is close to a source of gold, as the sound of counting money suggests a nearby NPC or monster with treasure.
This is not helpful and should be labelled as BAR.

Example 3: "5 gold pieces." indicates you collected 5 gold coins. This is not aligned with our goal and should be labelled as BAR.

Example 4: "That door is closed." is neutral and should be labelled as BAR.

Please rate this message: {{"{}"}}

Constraint: Please format your response strictly in this format without additional explanation:
<knowledge> [your knowledge of NetHack] </knowledge>
<analysis> [your one-sentence analysis of the message] </analysis>
<label> [FOO/BAR] </label>
""",
    "gold": """
Context: I am playing NetHack and I will present you a message from the gameplay.

Instruction: First, tell me about your knowledge of NetHack. Mention the goal of NetHack.
Then, write an analysis describing the semantics of each message using your knowledge of NetHack. Finally, respond by assigning a score to the message, which measures whether the status represented by the message is helpful or unhelpful to make some progress towards the goal.
Helpful here does not mean whether it is informative to the player to make decisions or not, but whether it is helpful for the agent to solve the game.
I prefer agents that maximize their gold. But never prefer agents that maximize the score in other ways (e.g., by engaging in combat or killing monsters) or that go down the dungeon.
For extremely helpful messages, assign a label FOO. For unhelpful, neutral, and lukewarm messages, assign a label BAR.

Example 1: "You kill the goblin!  Welcome to experience level 4" indicates that the player
has killed a goblin and gained experience.  This is not aligned with our goal and should be labelled as BAR.

Example 2: "You hear someone counting money." indicates that the player is close to a source of gold, as the sound of counting money suggests a nearby NPC or monster with treasure. This is extremely helpful and should be labelled as FOO.

Example 3: "5 gold pieces." indicates you collected 5 gold coins. This is extremely helpful and should be labelled as FOO.

Example 4: "That door is closed." is neutral and should be labelled as BAR.

Please rate this message: {{"{}"}}

Constraint: Please format your response strictly in this format without additional explanation:
<knowledge> [your knowledge of NetHack] </knowledge>
<analysis> [your one-sentence analysis of the message] </analysis>
<label> [FOO/BAR] </label>
""",
}


def generate_prompt_ranking(
    msg1, msg2, goal_key="defaultgoal", prompt_version="default"
):
    return ranking_prompt_templates[prompt_version].format(
        ranking_goal_strings[goal_key], msg1, msg2
    )


def get_ranking_goal_strings(goal_key):
    return ranking_goal_strings[goal_key]


def generate_prompt_classification(msg, goal_key="default_goal"):
    return classification_prompt_templates[goal_key].format(msg)
