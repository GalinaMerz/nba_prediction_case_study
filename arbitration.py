"""
Next-Best-Action arbitration layer.

Decision order is fixed and non-negotiable:

    1. ELIGIBILITY   — already holds the product        → drop candidate
    2. CONSENT       — no consented channel available    → drop candidate
    3. FREQUENCY     — contact cap already reached       → suppress all
    4. THRESHOLD     — P(accept) below floor             → drop candidate
    5. EXPECTED VALUE— rank survivors                    → choose one
    6. NO ACTION     — if nothing survives, or best EV<0 → suppress

A model score can never promote a candidate past steps 1-3. Rules are a
hard gate; ML ranks only what the gate allows through. This ordering is
what makes the decision auditable.

Values in ACTION_VALUE and CONTACT_COST are ASSUMPTIONS, not measured
figures. The dataset carries no revenue or contact-cost data. They are
here so expected-value arbitration can be demonstrated; any result that
depends on their exact magnitude is illustrative only.
"""

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------- assumptions

ACTION_VALUE = {
    "ind_cco_fin_ult1":  180.0,
    "ind_recibo_ult1":    40.0,
    "ind_nomina_ult1":   120.0,
    "ind_nom_pens_ult1": 110.0,
    "ind_ecue_fin_ult1":  60.0,
}

CONTACT_COST = {"inapp": 0.05, "push": 0.08, "email": 0.10, "outreach": 4.50}

# Channel preference order per action type. Servicing actions are
# time-sensitive and go to push/in-app; high-value offers may justify
# outreach. Rules-based by design: the dataset has no channel-response
# signal, so a learned channel model would be fitting noise (see framing
# doc s4.3).
CHANNEL_PLAN = {
    # time-sensitive servicing → push first
    "ind_recibo_ult1":   ["push", "inapp", "email"],
    # standard offers → email or in-app inbox (framing doc s4.3)
    "ind_ecue_fin_ult1": ["email", "inapp"],
    "ind_nomina_ult1":   ["email", "inapp", "outreach"],
    "ind_nom_pens_ult1": ["email", "inapp", "outreach"],
    # relationship anchor, higher value → in-app, then outreach
    "ind_cco_fin_ult1":  ["inapp", "email", "outreach"],
}

MIN_PROB = 0.01        # below this, a contact is not worth the attention cost
MAX_CONTACTS = 3       # per customer per period
HIGH_VALUE_EV = 25.0   # EV above which branch/contact-centre follow-up is flagged


# ---------------------------------------------------------------- data types

@dataclass
class Candidate:
    product: str
    score: float                    # calibrated P(accept)
    already_held: bool = False


@dataclass
class CustomerContext:
    customer_id: int
    consent: dict                   # {"email": bool, "push": bool, ...}
    contacts_this_period: int = 0


@dataclass
class Decision:
    customer_id: int
    action: Optional[str]           # None == no action
    channel: Optional[str] = None
    expected_value: Optional[float] = None
    probability: Optional[float] = None
    reason: str = ""
    needs_followup: bool = False
    trace: list = field(default_factory=list)   # audit trail, every step

    @property
    def suppressed(self) -> bool:
        return self.action is None


# ---------------------------------------------------------------- the gate

def _first_consented_channel(product, consent):
    for ch in CHANNEL_PLAN.get(product, ["inapp"]):
        if consent.get(ch, False):
            return ch
    return None


def arbitrate(ctx: CustomerContext, candidates: list,
              min_prob: float = MIN_PROB,
              max_contacts: int = MAX_CONTACTS) -> Decision:
    """Choose one action, or none, for a single customer."""
    trace = []

    # --- 3. FREQUENCY CAP (checked first: it suppresses everything) --------
    if ctx.contacts_this_period >= max_contacts:
        trace.append(f"frequency cap reached ({ctx.contacts_this_period}/{max_contacts})")
        return Decision(ctx.customer_id, None,
                        reason="Contact cap reached for this period.", trace=trace)

    survivors = []
    for c in candidates:
        # --- 1. ELIGIBILITY -----------------------------------------------
        if c.already_held:
            trace.append(f"{c.product}: dropped — already held")
            continue

        # --- 2. CONSENT ---------------------------------------------------
        channel = _first_consented_channel(c.product, ctx.consent)
        if channel is None:
            trace.append(f"{c.product}: dropped — no consented channel")
            continue

        # --- 4. THRESHOLD -------------------------------------------------
        if c.score < min_prob:
            trace.append(f"{c.product}: dropped — P={c.score:.4f} below floor {min_prob}")
            continue

        # --- 5. EXPECTED VALUE --------------------------------------------
        ev = c.score * ACTION_VALUE.get(c.product, 0.0) - CONTACT_COST[channel]
        trace.append(f"{c.product}: P={c.score:.4f} ch={channel} EV={ev:.2f}")
        survivors.append((ev, c, channel))

    # --- 6. NO ACTION ------------------------------------------------------
    if not survivors:
        return Decision(ctx.customer_id, None,
                        reason="No eligible, consented action clears the value floor.",
                        trace=trace)

    survivors.sort(key=lambda t: t[0], reverse=True)
    ev, best, channel = survivors[0]

    if ev <= 0:
        trace.append(f"best EV {ev:.2f} <= 0 — suppressing")
        return Decision(ctx.customer_id, None,
                        reason="Best available action has non-positive expected value.",
                        trace=trace)

    runner_up = survivors[1][0] if len(survivors) > 1 else None
    margin = "" if runner_up is None else f" (next best EV {runner_up:.2f})"

    return Decision(
        customer_id=ctx.customer_id,
        action=best.product,
        channel=channel,
        expected_value=ev,
        probability=best.score,
        reason=f"Highest expected value of {len(survivors)} permitted actions{margin}.",
        needs_followup=(ev >= HIGH_VALUE_EV),
        trace=trace,
    )


# ---------------------------------------------------------------- compliance

def consent_compliance_check(decisions, contexts) -> dict:
    """
    Must return violations == 0. A single violation blocks launch
    regardless of how the primary metric moved (framing doc s3).
    """
    violations = []
    ctx_by_id = {c.customer_id: c for c in contexts}
    for d in decisions:
        if d.suppressed:
            continue
        ctx = ctx_by_id[d.customer_id]
        if not ctx.consent.get(d.channel, False):
            violations.append((d.customer_id, d.action, d.channel))
        if ctx.contacts_this_period >= MAX_CONTACTS:
            violations.append((d.customer_id, d.action, "over-cap"))
    return {
        "contacted": sum(1 for d in decisions if not d.suppressed),
        "suppressed": sum(1 for d in decisions if d.suppressed),
        "violations": len(violations),
        "examples": violations[:5],
    }