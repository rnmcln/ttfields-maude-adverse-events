#!/usr/bin/env python3
"""Rule-based (regular expression) classifier of MAUDE TTFields narratives.

Category-specific patterns with a 40-character preceding-negation window and a fixed category
priority; severity, causality and device components by separate keyword rules.
Usage: python scripts/rule_based_classifier.py   (classifies data/derived/maude_ttfields_clean.csv and
compares the result with data/annotations/rule_based.csv)
"""
import re
import pandas as pd

# --- Pattern dictionaries (case-insensitive, applied to mdr_text_concat) ---

SKIN_PATTERNS = [
    r"\b(skin\s*(irritat|react|rash|erupt|break))",
    r"\b(dermatit|eczema|erythem|blister|ulcer(ation)?|lesion)",
    r"\b(wound\s*dehiscen|skin\s*tear|skin\s*peel)",
    r"\b(contact\s*(dermatit|allerg))",
    r"\b(redness|soreness|inflammation)\b.*\b(array|transducer|electrode|scalp)",
    r"\b(array|transducer|electrode).*\b(redness|soreness|inflammation)\b",
    r"\b(scalp\s*(burn|irritat|wound|sore|blister|reaction))",
    r"\b(skin\s*(breakdown|erosion|damage))",
]

INFECTION_PATTERNS = [
    r"\b(infect(ion|ed|ious))\b",
    r"\b(cellulitis|abscess|osteomyelit|meningit|sepsis|septic)\b",
    r"\b(wound\s*infect|surgical\s*site\s*infect)",
    r"\b(pus|purulent|drainage)\b",
    r"\b(antibiotic|antimicrobial).*\b(treat|prescri|administer)",
    r"\b(mrsa|staph|strep)\b",
]

HARDWARE_PATTERNS = [
    r"\b(malfunction|defect(ive)?|fail(ure|ed)?)\b",
    r"\b(battery|charger?|power\s*supply|adapter).*\b(fail|defect|issue|problem|stop|not\s*work)",
    r"\b(overheat|over-heat|thermal\s*(event|issue|shut))",
    r"\b(transducer|array|electrode).*\b(fail|defect|broken|damage)",
    r"\b(cable|cord|connector).*\b(fail|broken|damage|defect|fray)",
    r"\b(error\s*(code|message|display)|system\s*error)",
    r"\b(device\s*(stop|shut|not\s*work|ceased|inopera))",
    r"\b(short\s*circuit|electrical\s*(fault|issue))",
]

FALL_PATTERNS = [
    r"\b(fall|fell|trip(ped)?|stumbl)\b",
    r"\b(entangl|strangl|caught\s*in)\b.*\b(cable|cord|wire|device)",
    r"\b(cable|cord|wire|device).*\b(entangl|strangl|caught)",
    r"\b(pull(ed)?\s*(backward|down|over))\b.*\b(cord|cable|device)",
    r"\b(fracture|broken\s*bone).*\b(fall|fell)\b",
]

NEURO_PATTERNS = [
    r"\b(seizure|convuls|epilep)",
    r"\b(headache|cephalgia|migraine)\b",
    r"\b(cognitive\s*(declin|impair|dysfunction))",
    r"\b(confusion|disorientation|altered\s*mental)",
    r"\b(focal\s*(deficit|weakness|numbness))",
    r"\b(aphasia|dysphasia|speech\s*difficult)",
]

SYSTEMIC_PATTERNS = [
    r"\b(pulmonary\s*embol|pe\b|dvt\b|deep\s*vein\s*thromb)",
    r"\b(thrombo(sis|embol|tic))\b",
    r"\b(csf\s*leak|cerebrospinal\s*fluid\s*leak)",
    r"\b(intracranial\s*(hemorrh|bleed))",
    r"\b(hemorrhag|haemorrhag)\b",
    r"\b(pneumonia|respiratory\s*(fail|distress))\b",
    r"\b(cardiac\s*(arrest|event)|myocardial)",
    r"\b(renal\s*fail|kidney\s*fail)\b",
]

USER_ERROR_PATTERNS = [
    r"\b(incorrect(ly)?\s*(appl|plac|use))",
    r"\b(misus|improper\s*use|wrong\s*(array|placement|setting))",
    r"\b(non-?complian|poor\s*adherenc|did\s*not\s*(wear|use))",
    r"\b(user\s*error|patient\s*error|operator\s*error)",
    r"\b(array.{0,30}(wrong|incorrect|misplace|rotate))",
]

DISEASE_PROG_PATTERNS = [
    r"\b(disease\s*progress|tumor\s*(progress|recurr|grow))",
    r"\b(increased\s*tumor\s*size|mri\s*show.{0,30}(progress|growth|enlarg))",
    r"\b(clinical\s*deteriorat|neurologic(al)?\s*decline)\b",
    r"\b(death|died|expire|decease|fatal|pass(ed)?\s*away)",
]

# Negation detection: if a negation word precedes the match within 5 words, skip
NEGATION_WINDOW = 40  # characters
NEGATION_PATTERN = re.compile(
    r"\b(no|not|without|absence\s*of|denied|negative\s*for|ruled\s*out|"
    r"did\s*not|does\s*not|was\s*not|were\s*not|never)\b",
    re.IGNORECASE,
)


def _match_with_negation(text: str, patterns: list[str]) -> bool:
    """Return True if any pattern matches AND is not preceded by negation."""
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            start = max(0, m.start() - NEGATION_WINDOW)
            preceding = text[start : m.start()]
            if not NEGATION_PATTERN.search(preceding):
                return True
    return False


def _detect_severity(text: str, event_type: str) -> str:
    """Heuristic severity classification from text and MAUDE event_type."""
    text_lower = text.lower()
    if event_type == "Death" or re.search(r"\b(death|died|expire|fatal|decease)\b", text_lower):
        return "FATAL"
    if re.search(r"\b(hospitali[sz]|admitted|icu|intensive\s*care|surgery|surgical)\b", text_lower):
        return "SEVERE"
    if re.search(r"\b(treat(ed|ment)|medicat|antibiotic|prescri|outpatient|clinic\s*visit)\b", text_lower):
        return "MODERATE"
    return "MILD"


def _detect_causality(text: str) -> str:
    """Heuristic causality from text — conservative approach."""
    text_lower = text.lower()
    if re.search(r"\b(device\s*(caused|related|associated)|directly\s*(caused|due\s*to))\b", text_lower):
        return "DEFINITE"
    if re.search(r"\b(array|transducer|electrode|cable|optune).*\b(caused|resulted|led\s*to)\b", text_lower):
        return "PROBABLE"
    if re.search(r"\b(unclear|uncertain|unknown\s*(if|whether)|could\s*not\s*determin)\b", text_lower):
        return "UNASSESSABLE"
    if re.search(r"\b(not\s*related|unrelated|no\s*(relation|connection|association))\b", text_lower):
        return "UNLIKELY"
    return "POSSIBLE"  # default for ambiguous reports


def _detect_device_component(text: str) -> str:
    """Identify device component(s) mentioned."""
    text_lower = text.lower()
    components = []
    if re.search(r"\b(array|transducer|electrode|pad)\b", text_lower):
        components.append("ARRAY")
    if re.search(r"\b(cable|cord|wire|connector)\b", text_lower):
        components.append("CABLE")
    if re.search(r"\b(battery|charger|power\s*supply|adapter|charging)\b", text_lower):
        components.append("BATTERY")
    if re.search(r"\b(device\s*(unit|controller|box)|field\s*generator)\b", text_lower):
        components.append("UNIT")
    if re.search(r"\b(software|firmware|update|version)\b", text_lower):
        components.append("SOFTWARE")
    if not components:
        return "UNSPECIFIED"
    return "|".join(components)


def classify_regex(row: pd.Series) -> dict:
    """Classify a single MAUDE report using regex/rule-based approach."""
    text = str(row.get("mdr_text_concat", ""))
    event_type = str(row.get("event_type", ""))

    # Priority-ordered classification (first match wins for primary)
    primary = "UNCLASSIFIABLE"
    secondary = ""

    # Score each category
    scores = {
        "INFECTION": _match_with_negation(text, INFECTION_PATTERNS),
        "FALL_ENTANGLE": _match_with_negation(text, FALL_PATTERNS),
        "SYSTEMIC": _match_with_negation(text, SYSTEMIC_PATTERNS),
        "NEURO": _match_with_negation(text, NEURO_PATTERNS),
        "HARDWARE": _match_with_negation(text, HARDWARE_PATTERNS),
        "SKIN_NONINF": _match_with_negation(text, SKIN_PATTERNS),
        "USER_ERROR": _match_with_negation(text, USER_ERROR_PATTERNS),
        "DISEASE_PROG": _match_with_negation(text, DISEASE_PROG_PATTERNS),
    }

    matched = [cat for cat, hit in scores.items() if hit]

    if not matched:
        primary = "UNCLASSIFIABLE"
    elif len(matched) == 1:
        primary = matched[0]
    else:
        # Priority order for primary: INFECTION > SYSTEMIC > FALL > NEURO > HARDWARE > SKIN > USER_ERROR > DISEASE_PROG
        priority = [
            "INFECTION", "SYSTEMIC", "FALL_ENTANGLE", "NEURO",
            "HARDWARE", "SKIN_NONINF", "USER_ERROR", "DISEASE_PROG",
        ]
        for cat in priority:
            if cat in matched:
                primary = cat
                break
        # Secondary = second match
        remaining = [c for c in matched if c != primary]
        if remaining:
            for cat in priority:
                if cat in remaining:
                    secondary = cat
                    break

    severity = _detect_severity(text, event_type)
    causality = _detect_causality(text)
    device_component = _detect_device_component(text)

    return {
        "regex_primary_category": primary,
        "regex_severity": severity,
        "regex_causality": causality,
        "regex_device_component": device_component,
        "regex_secondary_category": secondary,
        "regex_notes": f"matched={','.join(matched)}" if matched else "no_match",
    }




def main():
    from paths import DERIVED, ANN
    df = pd.read_csv(DERIVED / "maude_ttfields_clean.csv", dtype={"report_number": str})
    res = df.apply(classify_regex, axis=1, result_type="expand")
    res.columns = [c.replace("regex_", "") for c in res.columns]
    res = res.rename(columns={"primary_category": "primary_category", "notes": "matched_categories"})
    out = pd.concat([df[["row_id", "report_number"]], res], axis=1)
    out.to_csv(DERIVED / "rule_based_rerun.csv", index=False)
    ref = pd.read_csv(ANN / "rule_based.csv", dtype={"report_number": str}).set_index("report_number").loc[out.report_number]
    for c in ["primary_category", "severity", "causality", "device_component", "secondary_category", "matched_categories"]:
        agree = (out[c].fillna("").astype(str).values == ref[c].fillna("").astype(str).values).mean()
        print(f"{c}: {100*agree:.1f}% identical to the archived rule-based classification")

if __name__ == "__main__":
    main()
