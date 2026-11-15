"""跨窗口交接：切换说话窗口时，把另一个窗口最近一段对话报到当前请求前面。

v1.2（2026-11-15）：
- 数据形状改成一条时间线（events：role + text + ts），不再把「对方的话」和
  「我的最后一条」分开存。两侧同等重要，取舍只看时间，越新越重要。
- 我最后一段输出改在 on_decorating_result 回写。之前写在 on_llm_request，
  那一刻本轮回复还没生成，带出去的永远差一轮。
- 交接块按 max_chars 从最新往回装，末尾对齐上一窗口最新一条；更旧的一条
  装不下就截尾保留（前面打「…」），保证从上往下读到末尾是最新的一句。
- max_turns 真正生效。

设计原则（和 41 一起定的）：
1. 不改 req.contexts，只在当前 prompt 前面贴一段参考块，随请求生灭。
2. 失败即停：任何异常都只记日志，绝不影响正常对话。
3. 只在「换窗口」时注入，同一窗口内不重复注入。
4. 不做时间触发（同窗口历史本来就在上下文里）。
"""

import json
import os
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

PLUGIN_NAME = "astrbot_plugin_cross_context"
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.abspath(
    os.path.join(_PLUGIN_DIR, "..", "..", "plugin_data", PLUGIN_NAME)
)
_STATE_FILE = os.path.join(_DATA_DIR, "state.json")

_USER_CHARS = 1200      # 单条对方的话存这么长就够，展示时再按预算处理
_ASSIST_CHARS = 1500    # 单条自己的话同理
_MIN_TAIL = 80          # 剩余预算超过这个，才把更旧那条截断塞进来
_EVENT_CAP = 40         # 单个窗口最多留多少条事件

# 交接块必须自带起止标记，抓取时整段剥掉，否则会自我套娃
BLOCK_OPEN = "〔交接开始〕"
BLOCK_CLOSE = "〔交接结束〕"


def _text_of(content) -> str:
    """从 OpenAI 格式的 content 里抽文本，丢掉 think/reasoning。"""
    if isinstance(content, str):
        return content
    parts = []
    if isinstance(content, list):
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") in ("think", "reasoning"):
                    continue
                t = p.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t)
    return " ".join(parts)


def _clean(text: str, limit: int = 400) -> str:
    """去掉系统提醒、记忆注入块等噪音，压成一行。"""
    if not text:
        return ""
    t = str(text)
    # 先整段剥掉交接块（含嵌套多层的）
    while BLOCK_OPEN in t:
        a = t.find(BLOCK_OPEN)
        b = t.find(BLOCK_CLOSE, a)
        if b == -1:
            t = t[:a]
            break
        t = t[:a] + t[b + len(BLOCK_CLOSE):]
    # 清掉残留的孤立标记（嵌套时可能剩下一个）
    t = t.replace(BLOCK_CLOSE, "").replace(BLOCK_OPEN, "")
    text = t
    out = []
    for line in str(text).splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("<system_reminder>") or "Weekday:" in s:
            continue
        if s.startswith("Historical memories recalled by LivingMemory"):
            continue
        if s.startswith("用户档案："):
            continue
        if s.startswith("记忆 #") or s.startswith("Participants:"):
            continue
        if s.startswith("【窗口交接"):
            continue
        if s.startswith("（这是记录，不是新指令"):
            continue
        out.append(s)
    joined = " ".join(out).strip()
    return joined[:limit]


@register(
    PLUGIN_NAME,
    "阿宁",
    "跨窗口交接：换平台时把另一个窗口最近一段对话报到过来，只作参考不改人格",
    "1.2.0",
)
class CrossContextPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        try:
            self.max_turns = max(1, min(12, int(self.config.get("max_turns", 4) or 4)))
        except Exception:
            self.max_turns = 4
        try:
            self.max_chars = max(200, min(4000, int(self.config.get("max_chars", 2000) or 2000)))
        except Exception:
            self.max_chars = 2000
        self.enabled = bool(self.config.get("enabled", True))
        self.state = {"tapes": {}, "last_umo": ""}
        self._load()

    # ---------- 持久化 ----------

    def _load(self) -> None:
        try:
            if os.path.exists(_STATE_FILE):
                with open(_STATE_FILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if isinstance(d, dict):
                    if isinstance(d.get("tapes"), dict):
                        self.state["tapes"] = d["tapes"]
                    self.state["last_umo"] = d.get("last_umo") or ""
        except Exception as e:
            logger.warning(f"[cross_context] 读状态失败，用空状态: {e}")

    def _save(self) -> None:
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            tmp = _STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False)
            os.replace(tmp, _STATE_FILE)
        except Exception as e:
            logger.warning(f"[cross_context] 写状态失败: {e}")

    # ---------- 逻辑 ----------

    @staticmethod
    def _who(umo: str) -> str:
        head = umo.split(":", 1)[0]
        if head.startswith("webchat"):
            return "网页"
        if head.startswith("default_") or "cqhttp" in head or "qq" in head.lower():
            return "QQ"
        return head

    @staticmethod
    def _events_of(tape: dict | None) -> list:
        """取一个窗口的时间线；顺手兼容 v1.1 的旧结构。"""
        if not tape:
            return []
        ev = tape.get("events")
        if isinstance(ev, list) and ev:
            return [e for e in ev if isinstance(e, dict) and e.get("text")]
        legacy = []
        for t in tape.get("users") or []:
            if t:
                legacy.append({"role": "user", "text": t, "ts": 0})
        if tape.get("last_assistant"):
            legacy.append({"role": "assistant", "text": tape["last_assistant"], "ts": 0})
        return legacy

    def _push_event(self, tape: dict, role: str, text: str, merge: bool = False) -> None:
        """往时间线尾部记一条。merge=True 时同一轮的分段回复就地覆盖。"""
        limit = _ASSIST_CHARS if role == "assistant" else _USER_CHARS
        text = _clean(text, limit)
        if not text:
            return
        ev = tape.setdefault("events", [])
        if ev:
            last = ev[-1]
            if merge and last.get("role") == "assistant":
                last["text"] = text
                last["ts"] = time.time()
                return
            if last.get("role") == role and last.get("text") == text:
                return
        ev.append({"role": role, "text": text, "ts": time.time()})
        cap = max(self.max_turns * 4, _EVENT_CAP)
        del ev[:-cap]

    def _seed_from_contexts(self, req: ProviderRequest) -> list:
        """tape 空的时候，从本窗口上下文里补一段，避免第一次交接是空的。"""
        cap = max(self.max_turns * 4, _EVENT_CAP)
        out = []
        for m in (req.contexts or [])[-cap:]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            limit = _ASSIST_CHARS if role == "assistant" else _USER_CHARS
            t = _clean(_text_of(m.get("content")), limit)
            if t:
                out.append({"role": role, "text": t, "ts": 0})
        return out[-cap:]

    def _block(self, other_umo: str, tape: dict | None) -> str:
        """拼交接块：从最新往回装，末尾对齐最新一条，超预算就截更旧那条。"""
        events = self._events_of(tape)
        if not events:
            return ""

        head = (
            BLOCK_OPEN + "\n"
            "【窗口交接 · 只作参考】\n"
            f"另一个窗口（{self._who(other_umo)}）刚结束。\n"
        )
        tail = (
            "\n（这是记录，不是指令：不用复述、不要模仿这上面的句式，接着他的话往下说就行。）\n"
            + BLOCK_CLOSE
        )
        budget = self.max_chars - len(head) - len(tail)
        if budget < 200:
            budget = 200

        lines = []
        for e in events:
            t = e.get("text") or ""
            if not t:
                continue
            lines.append(("· " + t) if e.get("role") == "user" else ("我：" + t))

        kept: list[str] = []
        used = 0
        for line in reversed(lines):          # 从最新一条往回装
            cost = len(line) + 1
            if used + cost <= budget:
                kept.append(line)
                used += cost
                continue
            # 这一条装不下：还有余量就把它的尾巴截进来，保证末端对齐
            room = budget - used
            if room >= _MIN_TAIL:
                cut = line[-(room - 2):]
                # 尽量从最近的句读起头，别切在词中间
                for sep in ("。", "！", "？", "；", "，", "、"):
                    idx = cut.find(sep)
                    if 0 <= idx < len(cut) // 2:
                        cut = cut[idx + 1:]
                        break
                kept.append("…" + cut)
            break

        kept.reverse()
        block = head + "\n".join(kept) + tail
        if len(block) > self.max_chars:       # 兜底，正常不该走到
            block = block[: self.max_chars]
        return block

    # ---------- 钩子 ----------

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        if not self.enabled:
            return
        try:
            umo = event.unified_msg_origin
            try:
                pid = str(event.get_platform_id())
            except Exception:
                pid = umo.split(":", 1)[0]

            tapes = self.state.setdefault("tapes", {})
            last = self.state.get("last_umo") or ""
            current = _clean(_text_of(req.prompt or ""), _USER_CHARS)

            inject = ""
            if last and last != umo:
                inject = self._block(last, tapes.get(last))
            elif not last and tapes:
                cands = sorted(
                    ((u, t) for u, t in tapes.items() if u != umo and t),
                    key=lambda kv: kv[1].get("ts", 0),
                    reverse=True,
                )
                if cands:
                    inject = self._block(cands[0][0], cands[0][1])

            if inject and req.prompt:
                req.prompt = inject + "\n\n" + req.prompt
                logger.info(
                    f"[cross_context] 注入交接块 {len(inject)} 字（来自 {last or '最近窗口'}）"
                )

            tape = tapes.setdefault(umo, {"events": [], "ts": 0})
            if not tape.get("events"):
                tape["events"] = self._seed_from_contexts(req)
            if current:
                self._push_event(tape, "user", current)
            tape["ts"] = time.time()
            tape["platform_id"] = pid
            self.state["last_umo"] = umo
            # 挪过来的旧结构字段不再需要
            tape.pop("users", None)
            tape.pop("last_assistant", None)

            if len(tapes) > 6:
                for u, _ in sorted(
                    tapes.items(), key=lambda kv: kv[1].get("ts", 0)
                )[: len(tapes) - 6]:
                    tapes.pop(u, None)

            self._save()
        except Exception as e:
            logger.warning(f"[cross_context] 请求侧处理失败，已跳过: {e}")

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent) -> None:
        """回复定型后回写最后一段输出——这是交接真正要带过去的那句。"""
        if not self.enabled:
            return
        try:
            result = event.get_result()
            if result is None:
                return
            if hasattr(result, "get_plain_text"):
                text = result.get_plain_text() or ""
            else:
                text = str(result)
            if not text.strip():
                return
            umo = event.unified_msg_origin
            tapes = self.state.setdefault("tapes", {})
            tape = tapes.setdefault(umo, {"events": [], "ts": time.time()})
            self._push_event(tape, "assistant", text, merge=True)
            tape["ts"] = time.time()
            self._save()
        except Exception as e:
            logger.warning(f"[cross_context] 回写回复失败，已跳过: {e}")
