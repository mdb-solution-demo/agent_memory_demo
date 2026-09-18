# -*- coding: utf-8 -*-
"""
配置文件模块（config.py）
========================
从环境变量或 .env 读取配置；集中存放提示词字符串。
供 main / memory_backends / db_init / npc_personas / custom_categories 使用。
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    root_env = Path(__file__).resolve().parents[2] / ".env"
    if root_env.exists():
        load_dotenv(root_env)
    else:
        load_dotenv()
except ImportError:
    pass


# --- 游戏主题 ---
GAME_THEME = "原神"

# --- 默认 NPC（agent id，须与 npc_personas 白名单 id 一致）---
NPC_NAME = "katheryne"

# --- 向量库 ---
MONGODB_URI = os.getenv("MONGODB_URI", "")
_mongodb_db_name = (os.getenv("MONGODB_DB_NAME") or "").strip()
if not _mongodb_db_name:
    raise ValueError(
        "必须在环境变量或项目根目录 .env 中配置 MONGODB_DB_NAME（无默认库名）"
    )
MONGODB_DB_NAME = _mongodb_db_name
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "extracted_memories")
EMBEDDING_DIMS = 1024

# --- game_wiki 混合检索（与 data_preparation 中集合/索引名对齐，可用环境变量覆盖）---
GAME_WIKI_ENABLED = os.getenv("GAME_WIKI_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)
# 未单独配置时与 Mem0 库同名；不再回退到硬编码库名
GAME_WIKI_DB_NAME = (os.getenv("GAME_WIKI_DB_NAME") or "").strip() or MONGODB_DB_NAME
GAME_WIKI_COLLECTION = os.getenv("GAME_WIKI_COLLECTION", "game_wiki")
GAME_WIKI_VECTOR_INDEX = os.getenv("GAME_WIKI_VECTOR_INDEX", "vector_search_index")
GAME_WIKI_TEXT_INDEX = os.getenv("GAME_WIKI_TEXT_INDEX", "text_search_index")
GAME_WIKI_RECALL_LIMIT = int(os.getenv("GAME_WIKI_RECALL_LIMIT", "3"))
GAME_WIKI_RECALL_MAX_CHARS = int(os.getenv("GAME_WIKI_RECALL_MAX_CHARS", "2000"))
GAME_WIKI_VECTOR_WEIGHT = float(os.getenv("GAME_WIKI_VECTOR_WEIGHT", "0.65"))
GAME_WIKI_SNIPPET_ANSWER_MAX = int(os.getenv("GAME_WIKI_SNIPPET_ANSWER_MAX", "1000"))

# $rankFusion 调试（与 rerank 无关）
GAME_WIKI_RANK_FUSION_SCORE_DETAILS = os.getenv(
    "GAME_WIKI_RANK_FUSION_SCORE_DETAILS", "false"
).strip().lower() in ("1", "true", "yes")

# --- NPC 多轮对话：请求携带的 conversation_history 截断（与 Mem0 retrieve 分离）---
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "40"))
CHAT_HISTORY_MAX_MSG_CHARS = int(os.getenv("CHAT_HISTORY_MAX_MSG_CHARS", "4096"))

# --- NPC 回复与 mem0：DeepSeek（OpenAI 兼容 API）---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# --- 嵌入（Voyage）---
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")


# =============================================================================
# 提示词（集中管理 - 原神主题）
# =============================================================================

MEMORY_METADATA_EXTRACT_SYSTEM_PROMPT_PREFIX = (
    "你是游戏对话「玩家状态」量表抽取器。根据【玩家】与【NPC】的对话进行分析，提取出玩家的状态信息，"
    "只输出一个 JSON 对象：键必须且仅能是下列维度名，每个键对应一个 0 到 1 之间的浮点数。\n"
    "数值格式：保留小数点后 1 位，例如 0.3、0.0、1.0、0.5；不要写成 0.35 或多位小数。\n"
    "语义约定：0 表示该维度上最差/最负面/最低，1 表示最好/最正面/最高；"
    "若无法从文本推断某维度，仍输出该键，取 0.5 表示中性。\n"
    "不要输出 null；不要输出说明文字；不要 markdown 或代码围栏。\n\n"
    "维度定义：\n"
)
MEMORY_METADATA_EXTRACT_SYSTEM_PROMPT_SUFFIX = "\n只输出 JSON。"

GENERIC_NPC_SYSTEM_PROMPT_TEMPLATE = (
    "你是开放世界游戏《原神》中的NPC角色「{name_zh}」（{name_en}）。\n"
    "你的说话方式应体现这些气质（不必在回复中提及 MBTI 或类型学术语）：{traits}。\n"
    "你需要根据与玩家的对话和「召回的记忆」并结合召回的[Wiki]片段进行自然回复；\n"
    "如果召回的[Wiki]是攻略相关到，则根据攻略内容结合用户实际记忆帮用户出一个切实的方案；\n"
    "若没有相关记忆，就依据当前对话与你的人设回答。\n"
    "回复简洁、口语化，符合角色身份。"
)

# 凯瑟琳等固定长人设提示词见 npc_personas.py（与 npc id 绑定）

CLASSIFY_SYSTEM = """你是一个游戏对话记忆分类器。根据本轮对话中提取出的玩家记忆事实（facts），判定它们应写入哪一级存储。

## 分类标准

### long_term（长期记忆 — 跨会话持久化）
玩家的**稳定属性和累积资产**，变化频率低，一旦确立在较长时间内保持不变。

判定条件（满足任意一条）：
- [身份] 类：冒险等阶、消费类型（月卡/零氪/大氪）、游戏时长、选择的旅行者
- [角色] 类：角色的拥有状态、命座等级（这些是不可逆的累积资产）
- [装备] 类：武器拥有状态（五星武器是永久资产）
- [偏好] 类：操作习惯偏好（简单/高难）、审美偏好、长期游戏目标类型
- 明确表达的**喜欢/讨厌**某角色、某玩法、某地区

### short_term（短期记忆 — 会话级或阶段性）
玩家的**当前状态和进行中的事项**，预期会在近期发生变化。

判定条件（满足任意一条）：
- [队伍] 类：当前使用的队伍配置（队伍经常调整）
- [装备] 类：当前正在使用的武器/圣遗物（会被替换）
- [进度] 类：正在推进的任务、深渊进度、当前刷本目标
- [困难] 类：当前遇到的卡关、资源短缺、操作困难
- [关注] 类：当前关注的角色、武器、圣遗物、队伍、深渊、本、活动等
- 正在进行的多步骤任务（"正在组队""正在刷圣遗物""在纠结选谁"）
- 近期计划（"准备抽xxx""打算刷xxx本"）

### zero_term（零记忆 — 不写入任何存储）
不包含任何玩家个人信息和游戏内容的对话。

判定条件（满足任意一条）：
- 提取结果为空 {"facts": []}
- 纯问候/寒暄/感谢/告别
- 对NPC回答的简单确认（"好的""知道了""原来如此"）

## 边界情况处理

以下情况需要特别注意：

1. **"刚抽到xxx"** → long_term（角色拥有是不可逆的永久事实）
2. **"正在练xxx"** → short_term（培养状态是阶段性的）
3. **"想抽xxx"** → short_term（意愿会变，且可能不会实现）
4. **"不玩xxx了/放弃了xxx"** → long_term（这是对已有角色使用状态的永久性更新）
5. **"深渊打到12层"** → short_term（进度在持续变化，下个周期可能不同）
6. **"月卡党"** → long_term（消费类型是稳定身份标签）
7. **"今天刷了20次绝缘本都没出好的"** → short_term（当前阶段的刷本状态）
8. **同一条 fact 含两类信息** → 取更高优先级（long_term > short_term > zero_term）

## 输入格式

你会收到本轮提取的 facts 列表（可能为空）。

## 输出格式

只回复一个词：long_term 或 short_term 或 zero_term。不要输出任何其他内容。

## 示例

Input: {"facts": []}
Output: zero_term

Input: {"facts": ["[角色] 用户刚抽到胡桃"]}
Output: long_term

Input: {"facts": ["[角色] 用户刚抽到胡桃", "[队伍] 用户想组建胡桃蒸发队"]}
Output: long_term

Input: {"facts": ["[进度] 用户深渊卡在12层", "[困难] 用户缺少风系聚怪角色"]}
Output: short_term

Input: {"facts": ["[身份] 用户冒险等阶55，月卡玩家"]}
Output: long_term

Input: {"facts": ["[装备] 用户当前胡桃装备匣里灭辰，没有护摩之杖"]}
Output: short_term

Input: {"facts": ["[关注] 用户关注圣遗物装备"]}
Output: short_term

Input: {"facts": ["[偏好] 用户偏好操作简单的角色"]}
Output: long_term

Input: {"facts": ["[角色] 用户选择了宵宫作为主力火C", "[角色] 用户暂时搁置胡桃"]}
Output: long_term

Input: {"facts": ["[困难] 用户觉得胡桃操作难度高，重击取消容易失误"]}
Output: short_term

Input: {"facts": ["[进度] 用户正在攒原石等芙宁娜复刻"]}
Output: short_term
"""

CUSTOM_FACT_EXTRACTION_PROMPT = """You are a memory extractor for a Genshin Impact intelligent NPC system. Your job is to extract ONLY persistent, personalizable facts from the PLAYER's messages that are useful for future conversations.

## EXTRACTION CATEGORIES (extract ONLY these types)

### A. Player Identity & Profile
- In-game name, nickname, AR (Adventure Rank), World Level
- Which Traveler they chose (Aether/Lumine)
- How long they've been playing, whether they're F2P or spending

### B. Character Ownership & Investment
- Characters the player OWNS, wants, or is building
- Constellation levels (e.g., "C2 Raiden", "C0 Hutao")
- Talent levels, friendship levels
- Characters the player explicitly DISLIKES or regrets pulling

### C. Equipment & Build Choices
- Weapons equipped or desired (e.g., "Staff of Homa on Hutao")
- Artifact sets being farmed or used (e.g., "4pc Crimson Witch")
- Specific stat priorities mentioned (e.g., "stacking EM on Kazuha")

### D. Team Composition & Playstyle
- Teams the player uses or is building (e.g., "Hutao vape team", "national team")
- Preferred playstyle: quickswap, hypercarry, comfort, meta, casual
- Difficulty preferences: easy rotation vs high-skill ceiling
- Spiral Abyss progress (e.g., "36-star Abyss", "stuck on Floor 12")

### E. Game Progress & Goals
- Current exploration region or Archon Quest progress
- Active goals (e.g., "saving primos for Furina rerun", "farming Emblem domain")
- Completed milestones (e.g., "finished Sumeru Archon Quest")
- Resin spending priorities

### F. Preferences & Opinions
- Favorite/least favorite characters, regions, bosses, or activities
- Opinions on game mechanics (e.g., "hate artifact RNG", "love exploration")
- Content preferences: story-focused, combat-focused, collection-focused

### G. Problems & Pain Points
- Difficulties the player is experiencing (e.g., "can't beat Azhdaha", "Hutao too hard to play")
- Resource shortages (e.g., "no good Pyro goblet", "out of mora")
- Frustrations expressed about specific content

---

## STRICT RULES

1. **[CRITICAL] Extract ONLY from the player's (user's) messages.** NEVER extract facts from the assistant/NPC responses, system messages, or knowledge base content. If the NPC says "Hutao uses Staff of Homa", that is NOT the player's memory.

2. **[CRITICAL] Ignore echoed knowledge.** If the player is merely repeating, confirming, or asking about general game knowledge (e.g., "what does vaporize do?", "how much resin for a domain?"), extract NOTHING — these are questions, not personal facts.

3. **Every fact MUST start with "用户"** as the subject. This is non-negotiable.

4. **Tag each fact with a category prefix** using this format:
   `[CATEGORY] 用户...`
   Valid prefixes: [身份], [角色], [装备], [队伍], [进度], [偏好], [困难]

5. **Be specific, not vague.** Bad: "用户在玩原神". Good: "用户拥有C1胡桃，正在练蒸发队".

6. **Merge related info into one fact** when they appear in the same message. Bad: two separate facts "用户有胡桃" + "用户胡桃是C1". Good: one fact "[角色] 用户拥有C1胡桃".

7. **Distinguish ownership from interest.** "我想抽胡桃" (wants) ≠ "我有胡桃" (owns). Use precise verbs: 拥有/已抽到 (owns), 想抽/计划抽 (wants), 在练/在培养 (building), 放弃了/不用了 (abandoned).

8. **Capture state changes.** If the player says they STOPPED using a character or CHANGED their team, extract the new state explicitly. E.g., "用户放弃了胡桃，改用宵宫作为火C".

9. **Do NOT extract:**
   - Greetings, pleasantries, filler ("hi", "thanks", "ok")
   - Questions that are purely informational with no personal context
   - Temporary/ephemeral states ("I'm logging off now", "let me think")
   - Anything the NPC/system said, even if the player quotes it

10. **Output language:** Match the player's input language. Chinese input → Chinese output. English input → English output. Mixed → follow the dominant language.

11. **Output format:** Return ONLY a single JSON object. No markdown, no code fences, no explanation.

---

## FEW-SHOT EXAMPLES

Input: 你好啊凯瑟琳
Output: {"facts": []}

Input: 今天的天气不错啊
Output: {"facts": []}
(Reason: purely informational question, no personal fact)

Input: 蒸发反应伤害怎么算？
Output: {"facts": [关注蒸发反应]}
(Reason: general game knowledge question)

Input: 对，我刚抽到胡桃，想组个蒸发队
Output: {"facts": ["[角色] 用户刚抽到胡桃", "[队伍] 用户想组建胡桃蒸发队"]}

Input: 我现在用的是匣里灭辰，还没有护摩之杖
Output: {"facts": ["[装备] 用户当前胡桃装备匣里灭辰，没有护摩之杖"]}

Input: 我冒险等阶55了，月卡党，璃月和蒙德都探索完了，正在打稻妻的主线
Output: {"facts": ["[身份] 用户冒险等阶55，月卡玩家", "[进度] 用户已完成璃月和蒙德探索，正在推进稻妻主线"]}

Input: 我感觉胡桃操作好难，重击取消好容易失误，想换个简单点的火C
Output: {"facts": ["[困难] 用户觉得胡桃操作难度高，重击取消容易失误", "[偏好] 用户偏好操作简单的角色，想更换火C"]}

Input: 我最后选了宵宫，感觉舒服多了，胡桃先放着吧
Output: {"facts": ["[角色] 用户选择了宵宫作为主力火C", "[角色] 用户暂时搁置胡桃"]}

Input: 深渊12层打不过去，缺一个风系角色聚怪
Output: {"facts": ["[进度] 用户深渊卡在12层", "[困难] 用户缺少风系聚怪角色"]}

Input: I have C2 Raiden, C1 Kazuha, and I'm saving for Furina's rerun
Output: {"facts": ["[角色] 用户拥有C2雷电将军和C1万叶", "[进度] 用户正在攒原石等芙宁娜复刻"]}

Input: 行秋和夜兰我都有，但是圣遗物一直刷不出好的绝缘套
Output: {"facts": ["[角色] 用户拥有行秋和夜兰", "[困难] 用户刷不到满意的绝缘之旗印套装"]}

Input: 好的，谢谢你的建议
Output: {"facts": []}

Input: 原来如此，那蒸发反应是火触发水乘以1.5倍对吧？
Output: {"facts": []}
(Reason: player is confirming learned knowledge, not stating personal info)
"""
