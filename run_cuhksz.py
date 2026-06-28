"""
多模型对话系统框架 
完整流程：
  1. Query Model    (32b) : 生成检索查询
  2. Retriever      (rag) : 调外部 RAG 检索 case
  3. 后处理         (4b)  :
       - SELECTION_LISTWISE  : 筛选相关 case
       - COMPRESS            : 单 case 句子级精简
       - KEYWORDS_LISTWISE   : 提取 keywords
  4. Trigger Model  (32b) : 判断是否触发知识
  5. Policy Model   (32b) : 决定对话动作
  6. Response Model (32b) : 生成回复

模型分工：
  - qwen3-4b   (端口 8800-8803, 通过 LLMRouter)        用于 selection / compress / keywords
  - qwen3-32b  (通过 qwen_api_multi.call_qwen_api)     用于 query / trigger / policy / response

"""

from json_repair import repair_json
import json
import re
import time
import sys
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from qwen_api_multi import call_qwen_api
from scu_rag import rag

# ==================== 日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 路径 ====================
INPUT_PATH = "/home/aarc/CuhkszTeam/nas1/RUNTIME/data_processing/案例推荐筛选/data/10/带caseID对话.json"
OUTPUT_PATH = "results/output.json"

# 取前多少条对话 (用于调试; None 表示全跑)
MAX_DIALOGUES = 300


# ==================== 后处理开关 ====================
SELECTION_ENABLED = True  ##rerank加selection
COMPRESS_ENABLED = False ##不需要用这个
KEYWORDS_ENABLED = True  ##提供额外信号（关键词）

# Selection 模式: "fixed" 或 "by_relevance"
SELECTION_MODE = "fixed"
SELECTION_FIXED_N = 5

# 重试 / 并发
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1
DIALOGUE_MAX_WORKERS = 12


# ==================== LLM (4b 后处理路由器) ====================
LLM_CONFIG = {
    "llm_list": [
        "http://localhost:8800/v1/chat/completions",
        "http://localhost:8801/v1/chat/completions",
        "http://localhost:8802/v1/chat/completions",
        "http://localhost:8803/v1/chat/completions",
    ],
    "model_name": "qwen3-4b",
    "temperature": 0.0,
    "timeout": 180,
}


class LLMRouter:
    """轮询调度本地 vLLM 实例 (4b)。线程安全。"""
    def __init__(self, llm_urls, model_name, temperature, timeout):
        self.llm_urls = llm_urls
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self._idx = 0
        self._lock = threading.Lock()

    def _next_url(self) -> str:
        with self._lock:
            url = self.llm_urls[self._idx]
            self._idx = (self._idx + 1) % len(self.llm_urls)
            return url

    def call(self, prompt: str) -> str:
        url = self._next_url()
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = requests.post(
            url, json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]


llm_router = LLMRouter(
    llm_urls=LLM_CONFIG["llm_list"],
    model_name=LLM_CONFIG["model_name"],
    temperature=LLM_CONFIG["temperature"],
    timeout=LLM_CONFIG["timeout"],
)


# ==================== 数据结构 ====================
@dataclass
class DialogueInput:
    current_turn: str
    dialogue_history: Any


@dataclass
class QueryResult:
    query: str = ""


@dataclass
class CaseDiagnostic:
    case_id: str
    total_sentences: int = 0
    kept_sentence_ids: List[str] = field(default_factory=list)
    kept_sentences: int = 0
    case_relevance: str = "partial"
    scenario_granularity: str = "generic"
    sid_case_content: str = ""
    keywords: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class RetrieverResult:
    evidence: str = "无"
    case_diagnostics: List[CaseDiagnostic] = field(default_factory=list)
    selection_relevances: Dict[str, str] = field(default_factory=dict)
    selection_failed: bool = False
    selected_case_ids: List[str] = field(default_factory=list)
    dropped_case_ids: List[str] = field(default_factory=list)


@dataclass
class TriggerResult:
    should_trigger: bool = False
    confidence: float = 0.0
    current_need: str = ""
    common_sense_judgement: str = ""
    case_assessment: List[Any] = field(default_factory=list)
    final_reasoning: str = ""


@dataclass
class PolicyResult:
    dialogue_action: str = ""
    case_id: str = ""
    reason: Any = ""


@dataclass
class ResponseResult:
    response: str = ""


@dataclass
class Latency:
    query_ms: int = 0
    retrieval_ms: int = 0
    selection_ms: int = 0
    compress_ms: int = 0
    keywords_ms: int = 0
    trigger_ms: int = 0
    policy_ms: int = 0
    response_ms: int = 0
    total_ms: int = 0


@dataclass
class PipelineOutput:
    dialogue_input: DialogueInput
    golden_response: str
    query_result: QueryResult
    retriever_result: RetrieverResult
    trigger_result: TriggerResult
    policy_result: PolicyResult
    response_result: ResponseResult
    latency: Latency


# ==================== Prompt 加载 ====================
with open("new_prompts/query.prompt", "r", encoding="utf-8") as f:
    query_prompt = f.read()
with open("new_prompts/trigger.prompt", "r", encoding="utf-8") as f:
    trigger_prompt = f.read()
with open("new_prompts/policy.prompt", "r", encoding="utf-8") as f:
    policy_prompt = f.read()
with open("new_prompts/response.prompt", "r", encoding="utf-8") as f:
    response_prompt = f.read()


# ==================== Prompt: SELECTION (listwise) ====================
SELECTION_LISTWISE_PROMPT = """# 角色

你是企业 IT 客服案例选择专家。检索阶段返回了 K 个候选案例,你需要根据当前对话状态,**对每一个候选案例**判断它与用户诉求的相关程度,从而帮助下游模型筛选出真正相关的案例。

# 背景

下游客服模型会从筛选后的案例池中选择一个推荐给用户。你的任务不是评判"案例本身写得好不好",而是判断"**案例的主题方向与用户当前诉求是否对齐,以及对齐的程度**"。

案例库的典型形态是**多子流程合辑**: 同一个案例可能同时覆盖申请、审批、查询、注销、延期等多个并列子流程。用户在对话中通常只触及其中一两个子话题。

# 输入

<dialogue_history>
{dialogue_history}
</dialogue_history>

<candidates>
{candidates_block}
</candidates>

# 评估标准

对每个候选案例,综合下列三方面给出 case_relevance:

1. **主题对齐**: 案例核心主题(从 title 和内容首句判断)与用户问的事项是否同一件事?
2. **子话题命中**: 用户提到的具体子话题(申请入口/审批/查询/注销/延期/异常处理等),案例是否覆盖?
3. **限定条件匹配**: 身份(华为员工/外协/WX)、平台(桌面端/手机端/内网/外网)、场景(临时/长期)等限定条件是否匹配?

## 评级规则

- `high`: 案例主题与对话完全对齐,且具体子话题命中,限定条件匹配。
- `partial`: 案例主题方向对齐,但只命中部分子话题,或限定条件部分匹配,或用户身份/场景未明确导致案例只是潜在适用。
- `low`: 案例主题与对话**不是同一件事**,仅因关键词或表面相似被召回。例: 用户问"外网权限",案例讲"端口权限(USB/网口物理端口)"; 用户问"员工自用WIFI",案例讲"访客WIFI"。

## 边界提醒

- **不要因为案例信息丰富/写得好就给 high**,只看与对话的对齐度。
- **不要因为关键词重叠就给 high**,要看实质是否同一事项。
- **partial 是常态**,不要吝啬给 partial。多数候选案例本身就处于 partial。
- **low 的判定要果断**,但需要有理由(主题确实不是同一件事,而非只是细节不全)。
- 主题方向只要对齐,即便案例覆盖的子话题用户没全部触及,也属于 partial 而非 low。
- 在并排扫描多个候选时,关注它们之间的差异——同一组里如果几个案例都看似相关,需要分辨哪个最契合(给 high),哪些只是部分相关(给 partial),哪些只是表面撞词(给 low)。

# 硬约束

1. 输出必须为合法 JSON,不得包裹 Markdown 代码块,不得有 JSON 之外的字符。
2. cases 数组的长度**必须与候选案例数完全一致**,每个候选案例都必须给出评级,不得遗漏。
3. 每个对象只输出 case_id 和 case_relevance 两个字段,不得输出其他字段。
4. case_id 必须严格从输入中复制,不得修改、伪造、重复。

# 输出格式

严格输出以下 JSON 对象:

{{
  "cases": [
    {{"case_id": "xxx", "case_relevance": "high" 或 "partial" 或 "low"}},
    ...
  ]
}}
"""


# ==================== Prompt: COMPRESS ====================
COMPRESS_PROMPT = """# 角色

你是企业 IT 客服案例后处理专家。负责对单个候选案例进行句子级精简。

# 背景

下游客服模型会同时看到多个候选案例,从中选一个最匹配当前对话的案例推荐给用户。你的精简结果会**替换原文**,直接送给下游模型。

案例库的典型形态是**多子流程合辑**:同一个案例往往同时覆盖"申请、审批、查询、延期、注销、适用对象分类、常见问题原因分析"等多个并列的子流程或子话题。用户在对话中通常只触及其中一两个。

你的工作不是判断"这句话写得好不好""信息量大不大",而是判断"**这句话所属的子话题或分支,是否服务于当前对话**"。

# 输入

<dialogue_history>
{dialogue_history}
</dialogue_history>

<case_title>
{case_title}
</case_title>

<case_content>
{sid_case_content}
</case_content>

# 总体原则: 保留为主,删除为辅

默认倾向**保留**,只在以下两类才删除:
- A 类: 无业务信息的纯结构性/装饰性句
- B 类: 与当前对话的子话题方向**明确无关**的分支段落

**不确定时的默认行为是保留**。

# 完整性底线(违反视为错误压缩)

**底线 1**: 若有解决方案/步骤/方法,至少一种完整方法的完整句群被保留。
**底线 2**: 保留问题现象句必须同时保留对应的原因/解决步骤。
**底线 3**: 结构锚点("一、申请入口""方法 1:" 等)与下属内容**配对保留或配对删除**。
**底线 4**: high 相关案例保留数明显多于删除数;若 high 但只剩 1-2 句,要么多保留要么下调到 partial。

# 逐句裁决

**问题 1: 纯结构/装饰句?**
属于以下任一种则删除(注意结构锚点例外见底线 3):
- 文档元信息式开场白、免责/提示套话、寒暄、工单痕迹
- 裸 URL 占位、空"P.S.:"、完全不承载业务信息的连接句

**问题 2: 与对话方向明确不一致?**
仅当用户问 A 该句讲 B、平台/场景明确不符、专门针对极少见边缘情形时删除; 其余一律保留(用户没点明子话题、案例多身份多类型、身份对照参考、同一子流程的相关说明、多个原因/方法、不同场景补充)。

**问题 3: 主题标识?**
开头 1-2 句主题陈述始终保留。保留主题标识后必须满足底线 2。

**问题 4: 完全重复?**
等价措辞重复,保留更完整易执行的那一句。

# low 情况

整个案例与对话**非同一事项**: 保留 2-4 句(主题标识 + 1-2 句最具代表性内容)。**不要只留 1 句**。

# 汇总

**case_relevance**: high(主题完全对齐) / partial(部分子话题对齐) / low(非同一事项)
**scenario_granularity**: specific / generic

**自洽性检查**:
- 若 high 但保留数 ≤ 2 → 重新审视
- 保留问题现象句但没保留任何解决方案句 → 违反底线 2
- 保留方法锚点但其下步骤全删 → 违反底线 3

# 硬约束

1. keep 中 ID 必须来自 case_content 中出现的 SID,不得伪造。
2. keep 至少含 1 个 SID。
3. keep 按原文 SID 出现顺序排列。
4. case_title 不参与编号。
5. 只能整句删除,严禁改写、截断、合并、润色。
6. 输出必须为合法 JSON,不得包含解释文本、Markdown 代码块标记。

# 输出格式

严格输出以下 JSON 对象:

{{
  "case_relevance": "high" 或 "partial" 或 "low",
  "scenario_granularity": "specific" 或 "generic",
  "keep": ["SID_x", "SID_y", ...]
}}
"""


# ==================== Prompt: KEYWORDS (listwise) ====================
KEYWORDS_LISTWISE_PROMPT = """# 角色

你是企业 IT 客服案例分析专家。负责为每个候选案例提取 keywords —— 案例的**内容签名**。

# 背景

下游客服模型(32B)会同时看到多个候选案例,从中选一个最匹配当前对话的案例推荐给用户。你的 keywords 会随每个案例一起呈现给下游,作为案例内容的**紧凑签名**,帮助下游模型在并排扫描时快速识别每个案例**讲的是什么**。

案例库的典型形态是**多子流程合辑**: 同一个案例往往同时覆盖申请、审批、查询、注销、延期等多个子流程。

# 任务定位 (重要)

你的**首要任务**是: 对每个案例,产出能**忠实概括该案例讲什么**的 1-3 个词。每一组 keywords 单独看,都应能让下游模型一眼看懂这个案例的内容方向。

你的**次要任务**是: 当案例池里多个案例主题相近时,可以让 keywords 中带有 1 个具区分作用的限定词,辅助下游分辨。这是**附加考虑**,不是强制要求。

**重要心智**:
- keywords 不是"找不同游戏"。如果两个案例本来就主题相同,它们的 keywords 可以重叠,**不必强行制造差异**。
- 区分力是 keywords **顺带产生**的属性,不是它的优化目标。一组写得忠实的内容签名,在大多数情况下天然就具有区分力。
- 你能同时看到所有候选,这是为了帮你做更准的内容概括(避免误读),而不是逼你必须从案例中挖出"独有特征"。

# 输入

<dialogue_history>
{dialogue_history}
</dialogue_history>

<cases>
{cases_block}
</cases>

# Keywords 提取规则

## 怎么算"内容签名"

合格的 keywords 应满足:
- **看得出主题**: 这个案例核心讲什么事(申请什么、解决什么问题、面向什么对象)
- **抓得住限定**: 案例里区别于"同类案例"的核心限定词(身份/平台/场景/系统),如果有
- **能在原文找到**: 关键实体词必须严格来自案例 case_content / case_title 字面

## 提取原则

**1. 先想"这个案例讲什么"**
读完一个案例,先用一句话在脑子里概括它的内容(例如"外协员工申请桌面端外网权限"),然后从这句话里摘出 2-3 个核心词作为 keywords。

**2. 再想"和池里相似案例怎么区分"(可选)**
扫一眼候选池里其他案例,如果有主题相同/相近的,看你刚才摘出的词够不够区分。
- 够 → 直接用,任务完成
- 不够 → 在 keywords 里替换/补一个能体现限定的词(身份/平台/场景/系统/版本)

如果池里没有主题相近的案例,跳过这一步。

**3. 字面来源严格**
- 关键实体(系统名/产品名/版本号/身份类型/功能名/业务术语): **必须严格来自原文 case_content / case_title 字面**。例: 原文"HarmonyOS NEXT"不能写成"鸿蒙系统"; 原文"外协员工"不能写成"非正式员工"。
- 动作语态(申请/恢复/排查/扩容/延期/注销): 若原文有现成精准词就用; 没有可在不歪曲案例实质的前提下浓缩。
- **不允许从 dialogue_history 借词**,不允许引入案例文本里完全没有的概念。

**4. 数量与长度**
- 每个案例 1-3 个 keywords,默认 2 个。
- 单个 keyword 长度 4-12 字。

## 反面示例 (避免)

❌ 只为了凸显差异而抽边角细节:
   案例讲"WIFI访客申请流程",抽成 [访客, 二维码] —— 漏了主题词"WIFI申请",下游看不出案例讲什么。

❌ 全用万能话题标签:
   多个案例都抽成 [权限申请, 流程指引] —— 没有任何信息量,等于没抽。

✅ 主题词 + 必要限定:
   案例讲"WIFI访客申请流程" → [访客WIFI, 申请流程]; 主题清晰,池里若有其他WIFI类案例时也能区分。

# 自洽性检查 (输出前逐条核对)

- [ ] 每组 keywords 单独看,是否能看出对应案例的内容方向?
- [ ] 关键实体词是否都能在对应案例原文中字面找到?
- [ ] 是否有从 dialogue_history 借词的情况?(不允许)
- [ ] 是否为了凸显差异而抽了与案例主题无关的边角细节?(应避免)

# 硬约束

1. 输出必须为合法 JSON,不得包裹 Markdown 代码块,不得有 JSON 之外的字符。
2. cases 数组的长度**必须与候选案例数完全一致**,每个案例都必须给出 keywords,不得遗漏。
3. 每个 case_id 必须严格从输入中复制,不得修改、伪造、重复。
4. 每个对象只输出 case_id 和 keywords 两个字段,不得输出其他字段。
5. 每个案例 keywords 数量 1-3 个,关键实体词必须字面来自 case_content/case_title。

# 输出格式

严格输出以下 JSON 对象:

{{
  "cases": [
    {{"case_id": "xxx", "keywords": ["关键词1", "关键词2"]}},
    ...
  ]
}}
"""


# ==================== 通用工具函数 ====================
def parse_json_response(text: Optional[str]) -> Optional[dict]:
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data[0] if data else None
        return data
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list):
                return data[0] if data else None
            return data
        except json.JSONDecodeError:
            pass
    try:
        data = json.loads(repair_json(text))
        if isinstance(data, list):
            return data[0] if data else None
        return data
    except Exception:
        return None


def normalize_rag_cases(rag_result: List[Dict]) -> List[Dict[str, str]]:
    """把 rag() 返回的 case list 归一化为 {case_id, title, content}。"""
    if not rag_result:
        return []
    cases = []
    for case in rag_result:
        cases.append({
            "case_id": str(case.get("case_id", "")),
            "title": str(case.get("title", "")),
            "content": str(case.get("content", "")),
        })
    return cases


def build_evidence_str(cases: List[Dict[str, Any]]) -> str:
    """构建 evidence 文本(case_id / title / [keywords] / content)。"""
    if not cases:
        return "无"
    evidence_str = ""
    for case in cases:
        evidence_str += f"case_id: {str(case.get('case_id', ''))}\n"
        evidence_str += f"title: {str(case.get('title', ''))}\n"
        kw = case.get("keywords", [])
        if kw and isinstance(kw, list) and len(kw) > 0:
            evidence_str += f"keywords: {', '.join(str(k) for k in kw)}\n"
        evidence_str += f"content: {str(case.get('content', ''))}\n\n"
    return evidence_str


def dialogue_history_to_str(dh: Any) -> str:
    if isinstance(dh, list):
        return "\n".join(str(x) for x in dh)
    return str(dh) if dh is not None else ""


# ==================== 句子预处理 (compress 用) ====================
def is_chinese_char(ch: str) -> bool:
    return bool(re.match(r'[\u4e00-\u9fff]', ch))


def preprocess_case_content_to_lines(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", " ").replace("\t", " ").replace("\n", " ")
    text = re.sub(r' +', ' ', text).strip()
    result = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch in "。!?;。!?;":
            result.append(ch)
            result.append("\n")
            continue
        if ch == " ":
            left_char = text[i - 1] if i - 1 >= 0 else ""
            right_char = text[i + 1] if i + 1 < n else ""
            if is_chinese_char(left_char) or is_chinese_char(right_char):
                result.append("\n")
            else:
                result.append(" ")
            continue
        result.append(ch)
    processed = "".join(result)
    processed = re.sub(r'\n+', '\n', processed)
    processed = "\n".join(line.strip() for line in processed.split("\n") if line.strip())
    return processed


def get_sentences(preprocessed_text: str) -> List[str]:
    return [line.strip() for line in preprocessed_text.split("\n") if line.strip()]


def build_sid_case_content(sentences: List[str]) -> Tuple[str, Dict[str, str], List[str]]:
    sid_to_text: Dict[str, str] = {}
    sid_list: List[str] = []
    lines: List[str] = []
    for i, sent in enumerate(sentences, start=1):
        sid = f"SID_{i}"
        sid_to_text[sid] = sent
        sid_list.append(sid)
        lines.append(f"<{sid}> {sent}")
    return "\n".join(lines), sid_to_text, sid_list


def sanitize_keep_ids(keep: Any, sid_list: List[str]) -> List[str]:
    if not isinstance(keep, list):
        return sid_list[:1] if sid_list else []
    valid_lower = {sid.lower(): sid for sid in sid_list}
    seen = set()
    cleaned: List[str] = []
    for x in keep:
        sid_l = str(x).strip().lower()
        if sid_l in valid_lower:
            real_sid = valid_lower[sid_l]
            if real_sid not in seen:
                cleaned.append(real_sid)
                seen.add(real_sid)
    if not cleaned and sid_list:
        cleaned = [sid_list[0]]
    sid_index = {sid: i for i, sid in enumerate(sid_list)}
    cleaned.sort(key=lambda x: sid_index[x])
    return cleaned


def sanitize_keywords(keywords: Any) -> List[str]:
    """规范化 keywords 列表: 去重、去空、限长 1-3 个"""
    if not isinstance(keywords, list):
        return []
    seen = set()
    cleaned: List[str] = []
    for k in keywords:
        s = str(k).strip()
        if s and s not in seen:
            cleaned.append(s)
            seen.add(s)
        if len(cleaned) >= 3:
            break
    return cleaned


def normalize_relevance(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in ("high", "h"):
        return "high"
    if s in ("partial", "p", "mid", "medium", "m"):
        return "partial"
    if s in ("low", "l"):
        return "low"
    return "partial"


def normalize_granularity(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in ("specific", "s"):
        return "specific"
    if s in ("generic", "g"):
        return "generic"
    return "generic"


# ==================== Selection (listwise) ====================
def call_selection_listwise(
    cases: List[Dict[str, str]],
    dialogue_history_str: str,
) -> Tuple[Dict[str, str], bool]:
    """
    Listwise selection: 一次输入全部 case,返回 case_id -> case_relevance。
    """
    if not cases:
        return {}, False

    candidates_lines = []
    for c in cases:
        candidates_lines.append(
            f"case_id: {c.get('case_id', '')}\n"
            f"title: {c.get('title', '')}\n"
            f"content: {c.get('content', '')}\n"
        )
    candidates_block = "\n---\n".join(candidates_lines)

    prompt = SELECTION_LISTWISE_PROMPT.format(
        dialogue_history=dialogue_history_str,
        candidates_block=candidates_block,
    )

    case_id_set = {c.get("case_id", "") for c in cases}

    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            text = llm_router.call(prompt)
            parsed = parse_json_response(text)
            if not parsed or "cases" not in parsed:
                logger.warning(f"[selection] attempt {attempt}: missing 'cases' field")
            else:
                items = parsed.get("cases") or []
                if isinstance(items, list) and len(items) > 0:
                    relevances: Dict[str, str] = {}
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        cid = str(it.get("case_id", "")).strip()
                        if cid in case_id_set:
                            relevances[cid] = normalize_relevance(it.get("case_relevance"))
                    if relevances:
                        for cid in case_id_set:
                            if cid not in relevances:
                                relevances[cid] = "partial"
                        return relevances, False
                logger.warning(f"[selection] attempt {attempt}: empty/invalid cases array")
        except Exception as e:
            logger.warning(f"[selection] attempt {attempt}: {type(e).__name__}: {e}")
        if attempt < RETRY_MAX_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS)

    logger.warning(f"[selection] all {RETRY_MAX_ATTEMPTS} attempts failed")
    return {}, True


def filter_cases_by_selection(
    cases: List[Dict[str, str]],
    relevances: Dict[str, str],
    mode: str,
    fixed_n: int,
) -> List[Dict[str, str]]:
    """根据 selection 结果筛选 case。"""
    if not relevances:
        return cases

    rank_map = {"high": 0, "partial": 1, "low": 2}
    rag_index = {c["case_id"]: i for i, c in enumerate(cases)}

    if mode == "fixed":
        cases_with_rel = [(c, relevances.get(c["case_id"], "partial")) for c in cases]
        cases_with_rel.sort(
            key=lambda x: (rank_map.get(x[1], 1), rag_index[x[0]["case_id"]])
        )
        return [c for c, _ in cases_with_rel[:fixed_n]]
    elif mode == "by_relevance":
        kept = [c for c in cases if relevances.get(c["case_id"], "partial") in ("high", "partial")]
        return kept
    else:
        raise ValueError(f"Unknown selection mode: {mode}")


# ==================== Compress (单 case 级并发) ====================
def compress_one_case(
    case: Dict[str, str],
    dialogue_history_str: str,
) -> Tuple[Dict[str, Any], CaseDiagnostic]:
    """单 case 句子级精简。失败时保留原文,不丢弃。"""
    case_id = case.get("case_id", "")
    title = case.get("title", "")
    content = case.get("content", "")

    if not content.strip():
        return {
            "case_id": case_id, "title": title, "content": content,
            "keywords": case.get("keywords", []),
        }, CaseDiagnostic(
            case_id=case_id, total_sentences=0, error="empty_content",
        )

    preprocessed = preprocess_case_content_to_lines(content)
    sentences = get_sentences(preprocessed)
    if not sentences:
        return {
            "case_id": case_id, "title": title, "content": content,
            "keywords": case.get("keywords", []),
        }, CaseDiagnostic(
            case_id=case_id, total_sentences=0, error="no_valid_sentences",
        )

    sid_case_content, sid_to_text, sid_list = build_sid_case_content(sentences)
    prompt = COMPRESS_PROMPT.format(
        dialogue_history=dialogue_history_str,
        case_title=title,
        sid_case_content=sid_case_content,
    )

    parsed = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            text = llm_router.call(prompt)
            parsed = parse_json_response(text)
            if parsed:
                break
            else:
                logger.warning(f"[compress][{case_id}] attempt {attempt}: parse failed")
        except Exception as e:
            logger.warning(f"[compress][{case_id}] attempt {attempt}: {type(e).__name__}: {e}")
        if attempt < RETRY_MAX_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS)

    if parsed is None:
        logger.warning(f"[compress][{case_id}] all attempts failed, keep original content")
        return {
            "case_id": case_id, "title": title, "content": content,
            "keywords": case.get("keywords", []),
        }, CaseDiagnostic(
            case_id=case_id,
            total_sentences=len(sentences),
            kept_sentence_ids=list(sid_list),
            kept_sentences=len(sid_list),
            sid_case_content=sid_case_content,
            error="compress_failed_keep_original",
        )

    case_relevance = normalize_relevance(parsed.get("case_relevance"))
    scenario_granularity = normalize_granularity(parsed.get("scenario_granularity"))
    keep = sanitize_keep_ids(parsed.get("keep", []), sid_list)
    kept_text = "\n".join(sid_to_text[s] for s in keep if s in sid_to_text)

    new_case = {
        "case_id": case_id,
        "title": title,
        "content": kept_text,
        "keywords": case.get("keywords", []),
    }
    diagnostic = CaseDiagnostic(
        case_id=case_id,
        total_sentences=len(sentences),
        kept_sentence_ids=keep,
        kept_sentences=len(keep),
        case_relevance=case_relevance,
        scenario_granularity=scenario_granularity,
        sid_case_content=sid_case_content,
    )
    return new_case, diagnostic


def compress_concurrent(
    cases: List[Dict[str, str]],
    dialogue_history_str: str,
) -> Tuple[List[Dict[str, Any]], List[CaseDiagnostic]]:
    """Compress 并发处理。"""
    if not cases:
        return [], []

    results_by_id: Dict[str, Tuple[Dict[str, Any], CaseDiagnostic]] = {}
    max_workers = max(1, len(cases))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cid = {
            executor.submit(compress_one_case, case, dialogue_history_str): case.get("case_id", "")
            for case in cases
        }
        for future in as_completed(future_to_cid):
            cid = future_to_cid[future]
            try:
                results_by_id[cid] = future.result()
            except Exception as e:
                logger.warning(f"[compress][{cid}] task exception: {e}, keep original")

    new_cases: List[Dict[str, Any]] = []
    diagnostics: List[CaseDiagnostic] = []
    for c in cases:
        cid = c.get("case_id", "")
        if cid in results_by_id:
            new_case, diag = results_by_id[cid]
            new_cases.append(new_case)
            diagnostics.append(diag)
        else:
            new_cases.append({
                "case_id": cid,
                "title": c.get("title", ""),
                "content": c.get("content", ""),
                "keywords": c.get("keywords", []),
            })
            diagnostics.append(CaseDiagnostic(
                case_id=cid, error="compress_task_exception_keep_original",
            ))
    return new_cases, diagnostics


# ==================== Keywords (listwise) ====================
def keywords_listwise(
    cases: List[Dict[str, Any]],
    dialogue_history_str: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Listwise keywords 提取。"""
    if not cases:
        return [], {}

    candidates_lines = []
    for c in cases:
        candidates_lines.append(
            f"case_id: {c.get('case_id', '')}\n"
            f"title: {c.get('title', '')}\n"
            f"content: {c.get('content', '')}\n"
        )
    candidates_block = "\n---\n".join(candidates_lines)

    prompt = KEYWORDS_LISTWISE_PROMPT.format(
        dialogue_history=dialogue_history_str,
        cases_block=candidates_block,
    )

    case_id_set = {c.get("case_id", "") for c in cases}

    parsed = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            text = llm_router.call(prompt)
            parsed = parse_json_response(text)
            if parsed and isinstance(parsed, dict) and "cases" in parsed:
                items = parsed.get("cases") or []
                if isinstance(items, list) and len(items) > 0:
                    break
                else:
                    logger.warning(f"[keywords] attempt {attempt}: empty cases array")
                    parsed = None
            else:
                logger.warning(f"[keywords] attempt {attempt}: missing 'cases' field")
                parsed = None
        except Exception as e:
            logger.warning(f"[keywords] attempt {attempt}: {type(e).__name__}: {e}")
            parsed = None
        if attempt < RETRY_MAX_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS)

    kw_map: Dict[str, List[str]] = {}

    if parsed is not None:
        items = parsed.get("cases") or []
        for it in items:
            if not isinstance(it, dict):
                continue
            cid = str(it.get("case_id", "")).strip()
            if cid in case_id_set:
                kw_map[cid] = sanitize_keywords(it.get("keywords", []))
    else:
        logger.warning(f"[keywords] all {RETRY_MAX_ATTEMPTS} attempts failed, keywords will be empty")

    new_cases: List[Dict[str, Any]] = []
    n_with_kw = 0
    for c in cases:
        cid = c.get("case_id", "")
        kws = kw_map.get(cid, [])
        if kws:
            n_with_kw += 1
        new_case = dict(c)
        new_case["keywords"] = kws
        new_cases.append(new_case)

    logger.info(f"[keywords] listwise: {n_with_kw}/{len(cases)} cases got keywords")
    return new_cases, kw_map


# ==================== 后处理总入口 ====================
def post_process_cases(
    raw_cases: List[Dict[str, str]],
    dialogue_history_str: str,
) -> Tuple[
    List[Dict[str, Any]], List[CaseDiagnostic], Dict[str, str], bool,
    List[str], List[str], int, int, int
]:
    """根据三个开关,对 raw_cases 做后处理。三模块独立 + 串行执行。"""
    final_cases: List[Dict[str, Any]] = []
    diagnostics: List[CaseDiagnostic] = []
    selection_relevances: Dict[str, str] = {}
    selection_failed = False
    selected_case_ids: List[str] = []
    dropped_case_ids: List[str] = []
    selection_ms = 0
    compress_ms = 0
    keywords_ms = 0

    if not raw_cases:
        return (final_cases, diagnostics, selection_relevances, selection_failed,
                selected_case_ids, dropped_case_ids, selection_ms, compress_ms, keywords_ms)

    # ---- Selection ----
    if SELECTION_ENABLED:
        sel_start = time.time()
        selection_relevances, selection_failed = call_selection_listwise(
            raw_cases, dialogue_history_str
        )
        selection_ms = int((time.time() - sel_start) * 1000)

        if selection_failed:
            cases_after_selection = list(raw_cases)
            selected_case_ids = []
            logger.warning(
                f"selection failed, fallback to all {len(raw_cases)} cases without filter"
            )
        else:
            cases_after_selection = filter_cases_by_selection(
                raw_cases, selection_relevances,
                mode=SELECTION_MODE, fixed_n=SELECTION_FIXED_N,
            )
            selected_case_ids = [c["case_id"] for c in cases_after_selection]
            n_high = sum(1 for v in selection_relevances.values() if v == "high")
            n_partial = sum(1 for v in selection_relevances.values() if v == "partial")
            n_low = sum(1 for v in selection_relevances.values() if v == "low")
            logger.info(
                f"selection: H={n_high} P={n_partial} L={n_low}, "
                f"selected={len(selected_case_ids)} (mode={SELECTION_MODE})"
            )
    else:
        cases_after_selection = list(raw_cases)
        selected_case_ids = []

    if not cases_after_selection:
        return (final_cases, diagnostics, selection_relevances, selection_failed,
                selected_case_ids, dropped_case_ids, selection_ms, compress_ms, keywords_ms)

    for c in cases_after_selection:
        c.setdefault("keywords", [])

    # ---- Compress ----
    if COMPRESS_ENABLED:
        comp_start = time.time()
        new_cases, comp_diags = compress_concurrent(
            cases_after_selection, dialogue_history_str,
        )
        compress_ms = int((time.time() - comp_start) * 1000)
        cases_after_compress = new_cases
        diagnostics = comp_diags
        n_compressed = sum(
            1 for d in comp_diags
            if not d.error and d.kept_sentences > 0
        )
        logger.info(
            f"compress: {n_compressed}/{len(cases_after_compress)} cases compressed "
            f"(others kept original due to error/empty)"
        )
    else:
        cases_after_compress = [
            {
                "case_id": c.get("case_id", ""),
                "title": c.get("title", ""),
                "content": c.get("content", ""),
                "keywords": c.get("keywords", []),
            }
            for c in cases_after_selection
        ]
        diagnostics = [
            CaseDiagnostic(
                case_id=c.get("case_id", ""),
                case_relevance=selection_relevances.get(c.get("case_id", ""), "partial"),
            )
            for c in cases_after_selection
        ]

    # ---- Keywords ----
    if KEYWORDS_ENABLED:
        kw_start = time.time()
        cases_after_keywords, kw_map = keywords_listwise(
            cases_after_compress, dialogue_history_str,
        )
        keywords_ms = int((time.time() - kw_start) * 1000)
        for d in diagnostics:
            d.keywords = kw_map.get(d.case_id, [])
    else:
        cases_after_keywords = cases_after_compress

    final_cases = cases_after_keywords

    return (final_cases, diagnostics, selection_relevances, selection_failed,
            selected_case_ids, dropped_case_ids, selection_ms, compress_ms, keywords_ms)


# ==================== Query / Retriever (新加,基于第一份代码) ====================
def call_query_model(dialogue_input: DialogueInput) -> Tuple[QueryResult, int]:
    """生成检索查询。使用 32b。"""
    start_time = time.time()
    messages = [{
        "role": "user",
        "content": query_prompt.replace("{p_dialogue}", str(dialogue_input))
    }]
    query = call_qwen_api(messages)
    elapsed_ms = int((time.time() - start_time) * 1000)
    return QueryResult(query=query or ""), elapsed_ms


def call_retriever(query: str, dialogue_history: List[str]) -> Tuple[List[Dict[str, str]], int]:
    """调用外部 RAG,返回归一化后的 case 列表。"""
    start_time = time.time()
    raw_evidence = rag(query, "\n".join(dialogue_history), "true")
    cases = normalize_rag_cases(raw_evidence or [])
    elapsed_ms = int((time.time() - start_time) * 1000)
    return cases, elapsed_ms


# ==================== Trigger / Policy / Response ====================
def call_trigger_model(evidence: str, dialogue_history_str: str) -> Tuple[TriggerResult, int]:
    start_time = time.time()
    messages = [{
        "role": "user",
        "content": trigger_prompt
            .replace("{messages}", dialogue_history_str)
            .replace("{retrieved_cases}", evidence)
    }]
    try:
        parsed = parse_json_response(call_qwen_api(messages))
        policy_mode = parsed.get("knowledge_support", "") if parsed else ""
    except Exception:
        parsed = {}
        policy_mode = ""
    if not policy_mode:
        policy_mode = "检索+常识"
    should_trigger = policy_mode in ["检索+常识", "检索 + 常识"]
    elapsed_ms = int((time.time() - start_time) * 1000)
    return TriggerResult(
        should_trigger=should_trigger,
        confidence=0.0,
        current_need=parsed.get("current_need", "") if parsed else "",
        common_sense_judgement=parsed.get("common_sense_judgement", "") if parsed else "",
        case_assessment=parsed.get("case_assessment", []) if parsed else [],
        final_reasoning=parsed.get("final_reasoning", "") if parsed else "",
    ), elapsed_ms


def call_policy_model(dialogue_history_str: str, evidence: str) -> Tuple[PolicyResult, int]:
    start_time = time.time()
    messages = [{
        "role": "user",
        "content": policy_prompt
            .replace("{dialog_history}", dialogue_history_str)
            .replace("{retrieved_cases}", evidence)
    }]
    parsed = parse_json_response(call_qwen_api(messages))
    dialogue_action = parsed.get("label", "") if parsed else ""
    case_id = parsed.get("case_id", "") if parsed else ""
    reason = parsed.get("reason", {}) if parsed else {}
    elapsed_ms = int((time.time() - start_time) * 1000)
    return PolicyResult(dialogue_action=dialogue_action, case_id=case_id, reason=reason), elapsed_ms


def call_response_model(
    dialogue_history_str: str, dialogue_action: str, evidence: str,
) -> Tuple[ResponseResult, int]:
    start_time = time.time()
    messages = [{
        "role": "user",
        "content": response_prompt
            .replace("{dialog_history}", dialogue_history_str)
            .replace("{retrieved_cases}", evidence)
            .replace("{dialogue_act}", dialogue_action)
    }]
    response = call_qwen_api(messages)
    elapsed_ms = int((time.time() - start_time) * 1000)
    return ResponseResult(response=response or ""), elapsed_ms


# ==================== 单 dialogue 处理 ====================
def _process_one(
    idx: int,
    dialogue_input: DialogueInput,
    golden_response: str,
    n: int,
) -> Tuple[int, Optional[PipelineOutput]]:
    entry_start = time.time()
    i = idx + 1
    try:
        dialogue_history = dialogue_input.dialogue_history
        dialogue_history_str = dialogue_history_to_str(dialogue_history)

        # 1. Query
        query_result, query_ms = call_query_model(dialogue_input)
        logger.info(f"[{i}/{n}] query generated: {query_result.query[:60]}...")

        # 2. Retriever (rag)
        raw_cases, retrieval_ms = call_retriever(
            query_result.query, dialogue_history if isinstance(dialogue_history, list) else []
        )
        logger.info(f"[{i}/{n}] retrieved {len(raw_cases)} case(s) from rag")

        # 3. 后处理 (selection + compress + keywords)
        (final_cases, diagnostics, selection_relevances, selection_failed,
         selected_case_ids, dropped_case_ids,
         selection_ms, compress_ms, keywords_ms) = \
            post_process_cases(raw_cases, dialogue_history_str)

        refined_evidence_str = build_evidence_str(final_cases) if final_cases else "无"

        rel_dist = {"high": 0, "partial": 0, "low": 0}
        for d in diagnostics:
            if d.case_relevance in rel_dist:
                rel_dist[d.case_relevance] += 1
        kw_summary = "; ".join(
            f"{d.case_id}:{d.keywords}" for d in diagnostics if d.keywords
        )
        logger.info(
            f"[{i}/{n}] post-process done: final={len(final_cases)} "
            f"H{rel_dist['high']}/P{rel_dist['partial']}/L{rel_dist['low']} "
            f"sel_ms={selection_ms} comp_ms={compress_ms} kw_ms={keywords_ms} "
            f"kw=[{kw_summary}]"
        )

        retriever_result = RetrieverResult(
            evidence=refined_evidence_str,
            case_diagnostics=diagnostics,
            selection_relevances=selection_relevances,
            selection_failed=selection_failed,
            selected_case_ids=selected_case_ids,
            dropped_case_ids=dropped_case_ids,
        )

        # 4. Trigger
        trigger_result, trigger_ms = call_trigger_model(
            refined_evidence_str, dialogue_history_str
        )
        logger.info(f"[{i}/{n}] trigger should_trigger={trigger_result.should_trigger}")

        # 5. Policy
        if trigger_result.should_trigger:
            policy_result, policy_ms = call_policy_model(
                dialogue_history_str, evidence=refined_evidence_str
            )
        else:
            policy_result, policy_ms = call_policy_model(
                dialogue_history_str, evidence="无"
            )
        logger.info(f"[{i}/{n}] policy.dialogue_action={policy_result.dialogue_action}")

        # 6. Response
        evidence_for_response = refined_evidence_str if trigger_result.should_trigger else "无"
        response_result, response_ms = call_response_model(
            dialogue_history_str,
            policy_result.dialogue_action,
            evidence=evidence_for_response,
        )
        logger.info(f"[{i}/{n}] response generated ({len(response_result.response)} chars)")

        total_ms = int((time.time() - entry_start) * 1000)

        pipeline_output = PipelineOutput(
            dialogue_input=dialogue_input,
            golden_response=golden_response,
            query_result=query_result,
            retriever_result=retriever_result,
            trigger_result=trigger_result,
            policy_result=policy_result,
            response_result=response_result,
            latency=Latency(
                query_ms=query_ms,
                retrieval_ms=retrieval_ms,
                selection_ms=selection_ms,
                compress_ms=compress_ms,
                keywords_ms=keywords_ms,
                trigger_ms=trigger_ms,
                policy_ms=policy_ms,
                response_ms=response_ms,
                total_ms=total_ms,
            ),
        )
        logger.info(f"[{i}/{n}] DONE in {total_ms}ms")
        return idx, pipeline_output

    except Exception as e:
        logger.error(f"[{i}/{n}] error: {e}", exc_info=True)
        return idx, None


# ==================== 主流程 (并发) ====================
def process_dialogue_parallel(
    dialogue_inputs: List[DialogueInput],
    golden_responses: List[str],
    max_workers: int = DIALOGUE_MAX_WORKERS,
) -> List[PipelineOutput]:
    total_start = time.time()
    n = len(dialogue_inputs)
    logger.info(
        f"Processing {n} dialogues, dialogue-parallel={max_workers}\n"
        f"  config: selection={SELECTION_ENABLED} compress={COMPRESS_ENABLED} keywords={KEYWORDS_ENABLED}"
    )
    if SELECTION_ENABLED:
        logger.info(f"  selection_mode={SELECTION_MODE} fixed_n={SELECTION_FIXED_N}")

    slots: List[Optional[PipelineOutput]] = [None] * n

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, idx, di, golden_responses[idx], n): idx
            for idx, di in enumerate(dialogue_inputs)
        }
        for future in as_completed(futures):
            idx, output = future.result()
            slots[idx] = output

    results = [r for r in slots if r is not None]
    failed = slots.count(None)
    total_elapsed = time.time() - total_start
    logger.info(
        f"All {n} dialogues processed in {total_elapsed:.2f}s, "
        f"got {len(results)} successful, {failed} failed"
    )
    return results


# ==================== 保存 ====================
def save_result(outputs: List[PipelineOutput], output_path: str = OUTPUT_PATH) -> None:
    result_list = []
    for i, output in enumerate(outputs, 1):
        result_dict = {
            "dialogue_id": i,
            "dialogue_input": asdict(output.dialogue_input),
            "golden_response": output.golden_response,
            "query": asdict(output.query_result),
            "retriever": asdict(output.retriever_result),
            "trigger": asdict(output.trigger_result),
            "policy": asdict(output.policy_result),
            "response": asdict(output.response_result),
            "latency": asdict(output.latency),
        }
        result_list.append(result_dict)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)
    logger.info(f"Result saved to {output_path}")


# ==================== 入口 ====================
def main():
    # 与第一份代码一致的数据加载方式: 从原始对话 JSON 直接构造样本
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        items = list(data.items())
        if MAX_DIALOGUES is not None:
            items = items[:MAX_DIALOGUES]

    dialogue_inputs: List[DialogueInput] = []
    golden_responses: List[str] = []

    for dialog_id, dialogue_content in items:
        dialogue_history: List[str] = []
        for turn in dialogue_content["text"]:
            if "用户" in turn:
                dialogue_history.append(f"用户: {turn['用户']}")
            else:
                current_turn = f"客服: {turn['客服']}"
                if "案例链接:" in current_turn:
                    dialogue_inputs.append(DialogueInput(
                        current_turn=current_turn,
                        dialogue_history=dialogue_history.copy(),
                    ))
                    golden_responses.append(current_turn)
                dialogue_history.append(current_turn)

    logger.info(f"Constructed {len(dialogue_inputs)} dialogue samples from {len(items)} dialogs")

    outputs = process_dialogue_parallel(dialogue_inputs, golden_responses)
    save_result(outputs)


if __name__ == "__main__":
    main()