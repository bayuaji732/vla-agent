from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from agent.config import get_settings
from agent.models import Action, ActionType, SafetyVerdict

logger = logging.getLogger(__name__)
cfg = get_settings()

# Keywords that indicate high-risk destructive actions
_HIGH_RISK_PATTERNS = re.compile(
    r"\b(delete|remove|cancel|unsubscribe|terminate|close account|"
    r"purchase|buy now|confirm order|pay|checkout|wire transfer|"
    r"send money|withdraw)\b",
    re.IGNORECASE,
)

_BLOCKED_URL_PATTERNS = re.compile(
    r"\b(" + "|".join(re.escape(d) for d in cfg.blocked_domains) + r")\b",
    re.IGNORECASE,
)


class SafetyGuard:
    """
    Pre-execution action validator.

    Risk levels:
      low     → execute immediately
      medium  → log warning, execute
      high    → require human approval if cfg.human_in_loop
      blocked → refuse unconditionally
    """

    async def check(self, action: Action, current_url: str = "") -> SafetyVerdict:
        verdict = self._evaluate(action, current_url)

        if verdict.risk_level == "blocked":
            logger.warning("BLOCKED action: %s | reason: %s", action.type, verdict.reason)
        elif verdict.risk_level == "high":
            logger.warning("HIGH-RISK action: %s | reason: %s", action.type, verdict.reason)
            if cfg.human_in_loop and not verdict.requires_human_approval:
                verdict = SafetyVerdict(
                    safe=False,
                    risk_level="high",
                    reason=verdict.reason,
                    requires_human_approval=True,
                )
        else:
            logger.debug("SAFE action: %s (%s)", action.type, verdict.risk_level)

        return verdict

    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate(self, action: Action, current_url: str) -> SafetyVerdict:
        # Terminal actions are always safe
        if action.type in (ActionType.DONE, ActionType.FAIL, ActionType.SCREENSHOT,
                           ActionType.WAIT, ActionType.SCROLL, ActionType.HOVER):
            return SafetyVerdict(safe=True, risk_level="low", reason="benign action type")

        # URL safety check
        if action.type == ActionType.NAVIGATE and action.url:
            if _BLOCKED_URL_PATTERNS.search(action.url):
                return SafetyVerdict(
                    safe=False,
                    risk_level="blocked",
                    reason=f"URL matches blocked domain pattern: {action.url}",
                )
            parsed = urlparse(action.url)
            if parsed.scheme not in ("http", "https", ""):
                return SafetyVerdict(
                    safe=False,
                    risk_level="blocked",
                    reason=f"Non-HTTP scheme blocked: {parsed.scheme}",
                )

        # Check typed text for dangerous patterns
        if action.type == ActionType.TYPE and action.text:
            if _HIGH_RISK_PATTERNS.search(action.text):
                return SafetyVerdict(
                    safe=False,
                    risk_level="high",
                    reason=f"Typed text matches high-risk pattern: '{action.text[:60]}'",
                    requires_human_approval=cfg.human_in_loop,
                )

        # Check rationale for dangerous intent
        if action.rationale and _HIGH_RISK_PATTERNS.search(action.rationale):
            return SafetyVerdict(
                safe=False,
                risk_level="high",
                reason=f"Action rationale indicates high-risk intent: '{action.rationale[:80]}'",
                requires_human_approval=cfg.human_in_loop,
            )

        return SafetyVerdict(safe=True, risk_level="low", reason="passed all checks")

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def request_human_approval(action: Action) -> bool:
        """
        Pause and ask the human operator to approve a risky action.
        Returns True if approved.
        In production this would integrate with a UI / Slack / etc.
        """
        print("\n" + "="*60)
        print("⚠️  HUMAN APPROVAL REQUIRED")
        print(f"   Action : {action.type}")
        print(f"   Target : elem={action.element_id} url={action.url} text={action.text!r}")
        print(f"   Reason : {action.rationale}")
        print("="*60)
        response = input("Approve? [y/N]: ").strip().lower()
        approved = response == "y"
        logger.info("Human approval: %s", "granted" if approved else "denied")
        return approved