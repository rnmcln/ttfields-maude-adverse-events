#!/usr/bin/env python3
"""Revised narrative descriptors (version 2), defined as non-exclusive report attributes.
Changes from version 1 (01_build_dataset.py):
- sentence-level co-occurrence: a site term and a lesion term must occur in the same sentence for
  surgical-site involvement; operative procedures must occur in a sentence without planning or
  pre-treatment wording;
- separate attributes for dehiscence, exposure, suspected and confirmed infection, bone-flap infection
  or osteomyelitis, and six classes of operative procedure;
- fall mechanism separated from injury mechanism; manufacturer statements that a contribution of the
  arrays could not be ruled out are recorded separately from statements of contribution;
- reporter attribution and death attribution taken from the full narrative, including manufacturer
  commentary; bevacizumab mention classified by context.
Two rule corrections were made after the language-model cross-check identified them (documented in the
supplement): indefinite articles and "broken" are accepted before array terms in statements of array
contribution to an injury, and "cause of death was not reported or unknown" is no longer read as an attribution.
Event attributes use the event narrative (manufacturer commentary sentences removed, as in version 1);
attribution and bevacizumab use the full narrative.
Inputs: analysis/data/analysis_dataset.csv, event_text_segments.csv, missed_reports.csv (if present).
Output: analysis/data/descriptors_v2.csv; analysis/data/descriptor_patterns_v2.json
"""
import re, json, pathlib
from paths import ROOT, DERIVED, ANN, RESULTS, TABLES, FIGURES, CODEBOOK
import pandas as pd

BOILER = re.compile(r"TRIAL|COMMERCIAL PROGRAM|PRESCRIBING INFORMATION|EXPECTED EVENT|INSTRUCTIONS FOR USE|"
                    r"NOVOCURE('S)? (MEDICAL )?OPINION|NOVOCURE AGREES|MEDICAL DEVICE REPORT|HDE#|CONTRIBUTING FACTOR|RISK FACTOR")
NEG = re.compile(r"\b(NO|NOT|WITHOUT|NEGATIVE FOR|DENIED|DENIES|RULED OUT|NOR|NEITHER)\b")
PLAN = re.compile(r"SCHEDULED|PLANN?ED|WOULD REQUIRE|MAY REQUIRE|MIGHT REQUIRE|CONSIDER|RECOMMEND|SUSPECTED THAT|WILL UNDERGO|"
                  r"TO BE PERFORMED|WAS DISCUSSED|OPTIONS?|IF NEEDED|POSSIBLE|CONSULT|(MIGHT|MAY|WOULD|COULD) BE (NEEDED|REQUIRED|NECESSARY)|DETERMINED|ADVISED|REFERR")
PRE = re.compile(r"PRIOR TO (STARTING|INITIATING|BEGINNING|COMMENCING|THE START OF)|BEFORE (STARTING|INITIATING|BEGINNING) (OPTUNE|THERAPY|TREATMENT|TTFIELDS)")

def split(t):
    return [s for s in re.split(r"(?<=[.!?])\s+|\s\|\s", str(t).upper()) if s.strip()]

def event_sents(t):
    return [s for s in split(t) if not BOILER.search(s)]

def hit(p, s, neg=False, win=40):
    for m in re.finditer(p, s):
        if neg and NEG.search(s[max(0, m.start() - win):m.start()]):
            continue
        return True
    return False

P = {
 "site": r"CRANIOTOMY|RESECTION (SITE|SCAR|INCISION|AREA|CAVITY)|SURGICAL (SITE|INCISION|WOUND|SCAR|RESECTION)|INCISION(?! AND DRAINAGE)|"
         r"\bSCARS?\b|BONE FLAP|SUTURE LINE|STAPLE LINE|OPERATIVE SITE|SHUNT",
 "lesion": r"ULCER|\bSORES?\b|WOUND|BREAK ?DOWN|DEHISC|EROSION|\bOPEN(ED|ING)?\b|LESION|INFECT|ABSCESS|\bPUS\b|PURULEN|DRAIN|LEAK|ERYTHEM|"
           r"REDNESS|SWELL|BLISTER|SCAB|IRRITAT|RASH|NECRO|EXPOS|PUSTUL|HEAL|DISCHARGE|BLEED|SKIN (DAMAGE|BREAKDOWN|ISSUE)",
 "placement": r"COULD NOT BE APPLIED|AVOID(ING)? THE|PLACE(D|MENT)? (THE )?ARRAYS? AWAY|ARRAY LAYOUT",
 "dehiscence": r"DEHISC|(INCISION|WOUND|SCAR|SUTURES?|STAPLE LINE)S? (HAD |HAS |WAS |WERE )?(RE)?(OPENED|OPENING|SEPARAT\w*|BROKE|BREAKING|SPLIT|GAPING)|"
               r"(WOUND|INCISIONAL|SCAR) BREAKDOWN|(OPENING|BREAKDOWN|SEPARATION|BREAK ?DOWN) (OF|AT|ALONG) (THE |HIS |HER )?(SURGICAL )?(INCISION|WOUND|SCAR)",
 "exposure": r"EXPOS\w*.{0,60}(HARDWARE|PLATE|SCREW|MESH|SHUNT|BONE|CATHETER|TUBING|CRANIUM|SKULL|DURA|IMPLANT|TITANIUM)|"
             r"(HARDWARE|PLATE|SCREW|MESH|SHUNT|TUBING|BONE|SKULL|IMPLANT|DURA)S? (WAS |WERE |IS |ARE |BECAME |BEING )?(VISIBLY |PARTIALLY )?(EXPOSED|VISIBLE)|"
             r"VISIBLE (BONE|SKULL|HARDWARE|PLATE|SCREW|MESH)",
 "infection_any": r"INFECT|CELLULITIS|OSTEOMYELITIS|ABSCESS|\bPUS\b|PURULEN|ANTIBIOTIC|EMPYEMA|MENINGITIS|SEPTIC",
 "infection_confirmed": r"STAPH|STREPT|PROPIONI|CUTIBACT|\bMRSA\b|\bMSSA\b|PSEUDOMON|ENTEROCOC|E\.? ?COLI|SERRATIA|KLEBSIEL|ENTEROBACT|PROTEUS|"
                        r"CORYNEBACT|ACINETOBACTER|CANDIDA|CULTURES? (WAS |WERE |CAME BACK |RETURNED )?POSITIVE|POSITIVE (WOUND |BLOOD )?CULTURE|"
                        r"CULTURES? (DID )?(GR[EO]W|GROWING|REVEALED|SHOWED|IDENTIFIED)|\bPUS\b|PURULEN|ABSCESS|EMPYEMA",
 "culture_negative": r"NO GROWTH|CULTURES? (WAS |WERE |CAME BACK |RETURNED )?NEGATIVE|NEGATIVE CULTURE",
 "bone_infection": r"OSTEOMYELITIS|BONE FLAP INFECT|INFECTED (BONE|SKULL|BONE FLAP|FLAP|HARDWARE|PLATE|MESH)|"
                   r"INFECTION OF THE (BONE|SKULL|BONE FLAP|FLAP|HARDWARE|PLATE|MESH|CRANIOPLASTY)|(BONE|SKULL|FLAP|HARDWARE|MESH|PLATE) (WAS |WERE )?INFECTED",
 "iv_antibiotics": r"\bIV ANTIBIOTIC|INTRAVENOUS (ANTIBIOTIC|VANCOMYCIN|CEF)|VANCOMYCIN|CEFAZOLIN|CEFTRIAXONE|MEROPENEM|CEFEPIME|PIPERACILLIN|"
                   r"NAFCILLIN|DAPTOMYCIN|\bPICC\b|OXACILLIN|LINEZOLID",
 "hospitalised": r"HOSPITALI[SZ]|\bADMITTED\b|ADMISSION|INPATIENT",
 "temporary_interruption": r"TEMPORARILY (DISCONTINU|INTERRUPT|STOPP|PAUS|SUSPEND|HELD|HALT)|THERAPY BREAK|TREATMENT BREAK|\bPAUSED\b|"
                           r"PLANN?ED TO RESUME|RESUMED (OPTUNE|THERAPY|TREATMENT|DEVICE|USE)|RESTARTED (OPTUNE|THERAPY|TREATMENT)",
 "permanent_discontinuation": r"PERMANENTLY (DISCONTINU|STOPP)|DISCONTINUED [A-Z ]{0,30}PERMANENTLY|WILL NOT RESUME|DID NOT RESUME|"
                              r"DECIDED TO (STOP|DISCONTINUE|END)|NO LONGER (ON|USING|RECEIVING) (OPTUNE|THERAPY|THE DEVICE)",
}
PROC = {  # operative procedure classes (event sentences without planning or pre-treatment wording)
 "debridement_or_washout": r"DEBRID|WASH ?OUT|IRRIGATION AND DEBRIDEMENT|\bI ?& ?D\b|CLEAN-?OUT|(PROCEDURES?|SURGER\w*) TO CLEAN",
 "incision_and_drainage": r"INCISION AND DRAINAGE|SURGICAL(LY)? DRAIN|DRAINAGE OF (THE |AN |A )?(\w+ )?(ABSCESS|WOUND|FLUID|COLLECTION|EMPYEMA)",
 "bone_flap_removal": r"BONE FLAP.{0,40}(REMOV|EXCIS|EXPLANT|RESECT)|(REMOV|EXCIS|EXPLANT)\w*.{0,40}BONE FLAP|CRANIECTOMY|"
                      r"(PART|PORTION|PIECE) OF (THE |HIS |HER )?(SKULL|BONE).{0,30}(REMOV|EXCIS)|(SKULL|BONE) (WAS |WERE )?REMOVED",
 "hardware_or_mesh_removal": r"(HARDWARE|PLATES?|SCREWS?|MESH|IMPLANT|TITANIUM|CRANIOPLASTY).{0,30}(WAS |WERE )?(REMOV|EXPLANT|REPLAC)|"
                             r"(REMOV|EXPLANT)\w*.{0,40}(HARDWARE|PLATE|SCREW|MESH|IMPLANT|TITANIUM)",
 "shunt_revision_or_removal": r"SHUNT (REVISION|REMOV|EXTERNALI|REPLACE)|(REVISION|REMOVAL|REPLACEMENT|EXTERNALI[SZ]ATION) OF (THE |HIS |HER )?\w* ?SHUNT|"
                              r"SHUNT (WAS |WERE )?(REVISED|REMOVED|REPLACED|EXTERNALI)",
 "flap_or_graft_reconstruction": r"(ROTATION(AL)?|ADVANCEMENT|FREE|LOCAL|MUSCLE|TISSUE|SCALP|SKIN) FLAP|SKIN GRAFT|GRAFT(ING)?\b|"
                                 r"FLAP (CLOSURE|RECONSTRUCTION|COVERAGE|SURGERY)|PLASTIC SURG|RECONSTRUCT",
 "wound_revision_or_closure": r"WOUND REVISION|CRANIOTOMY REVISION|REVISION CRANIOTOMY|REVISION (SURGERY|OF THE (WOUND|INCISION|SCALP|FLAP|SCAR))|(SCALP|WOUND|INCISION|SCAR) (SURGERY|REPAIR|REVISION|CLOSURE)|"
                              r"(SURGERY|SURGICALLY|OPERATION|PROCEDURE)\w*.{0,60}(TO )?(CLOSE|CLOSURE|REPAIR)|SURGICALLY (CLOSED|REPAIRED|EXCISED|REMOVED)|"
                              r"RESUTUR|RE-?SUTUR|(WAS|WERE) (RE)?SUTURED (CLOSED|SHUT)",
 "unspecified_surgery": r"(UNDERWENT|REQUIRED|REQUIRING|HAD|UNDERGO|UNDERGOING|NEEDED|PERFORMED)\b.{0,40}\b(SURGERY|SURGERIES|SURGICAL (INTERVENTION|PROCEDURE)|OPERATION|"
                        r"PROCEDURE|REOPERATION|RE-OPERATION)\b.{0,80}(INFECT|WOUND|DEHISC|ULCER|SCALP|INCISION|SCAR|ABSCESS|EXPOS|LESION|SKIN)|"
                        r"(INFECT|WOUND|DEHISC|ULCER|ABSCESS|EXPOS)\w*.{0,60}(REQUIRED|REQUIRING|NECESSITAT)\w* .{0,20}(SURGERY|SURGICAL|OPERATION)",
}
FALL = {
 "tripped_or_entangled_in_cable": (r"TRIP|CAUGHT|ENTANGL|TANGL|HOOKED|STEPPED ON|WRAPPED", r"CABLE|\bCORDS?\b|\bWIRES?\b|CONNECTION"),
 "pulled_by_cable": (r"\bPULL|YANK|TUG|JERK", r"CABLE|\bCORDS?\b|\bWIRES?\b|POWER SUPPLY|PLUGGED|WALL"),
 "carrying_device": (r"\bBAG\b|BACKPACK|STRAP|\bWEIGHT\b|HEAVY|CARRY|CARRIED|CARRYING|SHOULDER", r"FELL|\bFALL|BALANCE|TRIP"),
 "other_cause_stated": (r"SEIZURE|SYNCOP|FAINT|PASSED OUT|DIZZ|WEAK|LOST CONSCIOUSNESS|LOSS OF CONSCIOUSNESS|SLIPPED|LOST (HIS|HER) BALANCE|"
                        r"LOSS OF BALANCE|HEMIPARESIS|STROKE|GAIT|LEG GAVE|KNEES? (GAVE|BUCKLED)|UNSTEADY|LIGHTHEADED|LIGHT-HEADED|ORTHOSTATIC|HYPOTENSI", r"FELL|\bFALL|COLLAPS"),
}
FALL_INJ = {
 "scalp_or_head_wound": r"LACERAT|HEAD (INJURY|TRAUMA|WOUND)|SCALP (WOUND|INJURY|CUT)|SUTURE[DS]|STAPLED|STAPLES|CONTUSION|ABRASION|GASH|"
                        r"CUTS? (ON|TO) (HIS|HER|THE) (HEAD|SCALP|FOREHEAD)|BUMP ON (HIS|HER|THE) HEAD|HEMATOMA ON|WOUNDS? ON (HIS|HER|THE) HEAD",
 "fracture": r"FRACTUR",
 "intracranial_haemorrhage": r"SUBDURAL|EPIDURAL H|SUBARACHNOID|INTRACRANIAL (HEMORRHAGE|HAEMORRHAGE|BLEED)|INTRAPARENCHYMAL|INTRACEREBRAL|"
                             r"BRAIN (BLEED|HEMORRHAGE)|(INTRACRANIAL|CEREBRAL|SUBDURAL|EPIDURAL) (HEMATOMA|HAEMATOMA)",
}
ARR = r"ARRAYS?|TRANSDUCER|DIS[CK]S?|CERAMIC|CONNECTOR"
DEV = r"ARRAYS?|TRANSDUCER|DISCS?|DISKS?|CERAMIC|CONNECTOR|\bDEVICE\b|OPTUNE|CABLE|\bCORD\b"
INJ = r"LACERAT|\bCUT\b|WOUND|INJUR|ABRASION|SKIN|HEMATOMA|FRACTUR|BRUIS"
CRBO = r"(CANNOT|COULD NOT|CAN NOT|CAN'T|COULDN'T) BE (RULED OUT|EXCLUDED)|NOT BE (RULED OUT|EXCLUDED)"
REL_POS = (r"(RELATED TO|CAUSED BY|DUE TO|ASSOCIATED WITH|ATTRIBUTED TO|SECONDARY TO|RESULT OF|FROM) (THE )?(USE OF )?(OPTUNE|DEVICE|TRANSDUCER ARRAYS?|ARRAYS?|"
           r"TTFIELDS|NOVOTTF|THERAPY WITH OPTUNE|OPTUNE (GIO|LUA)? ?THERAPY)|(OPTUNE|DEVICE|ARRAY|TRANSDUCER ARRAY|TTFIELDS)[- ](RELATED|INDUCED|ASSOCIATED)")
REL_NEG = (r"(NOT|UN)\s?(RELATED|CAUSED|DUE|ASSOCIATED|ATTRIBUT)\w*\s(TO|BY|WITH)\s(THE )?(USE OF )?(OPTUNE|DEVICE|TRANSDUCER|ARRAY|TTFIELDS|NOVOTTF|THERAPY)|"
           r"NO (CAUSAL )?(RELATIONSHIP|ASSOCIATION|CONTRIBUTION).{0,40}(OPTUNE|DEVICE|ARRAY|TTFIELDS|NOVOTTF)")
MECH = (r"(UNDER|BENEATH|BELOW) (THE )?(TRANSDUCER )?ARRAYS?|ARRAY (PLACEMENT|ADHESIVE|SITE|EDGE|GEL|LOCATION|AREA|TAPE)|ADHESIVE|"
        r"PRESSURE (FROM|OF|ULCER)|(HEAT|HOT|WARM|BURN)\w*|SHAV\w*|(TRANSDUCER )?ARRAYS? (WERE |WAS )?(PLACED|APPLIED|POSITIONED) (OVER|ON)|"
        r"CABLE|\bCORDS?\b|ENTANGL|MEDICAL DEVICE SITE|DEVICE SITE|HYDROGEL|\bGEL\b|CONNECTOR|TAPE")
DEATH = r"\bDIED\b|PASSED AWAY|\bEXPIRED\b|\bDECEASED\b|DATE OF DEATH|(PATIENT|PT|HIS|HER)'?S? DEATH|DEATH (OF THE PATIENT|OCCURRED|WAS REPORTED|ON)|\bDEATH\b"
DEATH_GENERIC = r"EXPECTED EVENT|TRIAL|SURVIVAL|NATURAL HISTORY OF|PRESCRIBING INFORMATION|INCIDENCE|RISK OF"
D_OTHER = (r"(DEATH|DIED|PASSED AWAY|EXPIRED)\w*.{0,120}(DUE TO|FROM|RELATED TO|SECONDARY TO|ATTRIBUTED TO|CAUSED BY|RESULT OF|COMPLICATIONS? OF|CAUSE OF DEATH)"
           r".{0,60}(PROGRESS|TUMOU?R|DISEASE|GBM|GLIOBLASTOMA|GLIOSARCOMA|GLIOMA|CANCER|PNEUMONIA|SEPSIS|STROKE|EMBOL|HEMORRHAGE|HAEMORRHAGE|MALIGNAN|"
           r"MESOTHELIOMA|RESPIRATORY|CARDIAC|INFARCT|ASPIRATION|HERNIATION|EDEMA)|CAUSE OF DEATH (WAS|WAS REPORTED AS|:)(?! ?(NOT|UNK|UNKNOWN|NO ))|"
           r"(DEATH|IT) (WAS|IS|IS CONSIDERED) (NOT RELATED|UNRELATED)|DEATH IS UNRELATED|DEATH WAS UNRELATED|(DEATH|DIED).{0,60}NOT RELATED TO (OPTUNE|THE DEVICE|NOVOTTF|TTFIELDS)")
D_DEV = (r"CONTRIBUTION OF (THE )?(DEVICE|OPTUNE|NOVOTTF|ARRAYS?|TTFIELDS|THERAPY).{0,40}DEATH.{0,40}(" + CRBO + r")|"
         r"DEATH.{0,80}(POSSIBLY|PROBABLY|MAY HAVE BEEN|LIKELY) (RELATED|DUE|CAUSED)")
BEV = r"BEVACIZUMAB|AVASTIN|MVASI|ZIRABEV"
BEV_NOT = r"(NOT|NEVER) (BEEN )?(ON|TAKING|RECEIVING|RECEIVED|TREATED WITH|TAKEN|USED|ADMINISTERED)\b.{0,30}(" + BEV + r")|\bNO (PRIOR |CONCOMITANT |PREVIOUS )?(" + BEV + r")|(" + BEV + r")[- ]NAIVE|WITHOUT (" + BEV + r")"
BEV_PREV = (r"(PRIOR|PREVIOUS|PREVIOUSLY|HISTORY OF|FORMERLY|HAD RECEIVED|HAD BEEN ON|DISCONTINUED|STOPPED|PRETREATMENT|PRE-TREATMENT|PRETREATED|COMPLETED)\b.{0,40}(" + BEV + r")|"
            r"(" + BEV + r").{0,40}(PREVIOUSLY|IN THE PAST|WAS DISCONTINUED|WAS STOPPED|HAD BEEN DISCONTINUED|HAD BEEN STOPPED|ENDED)")
BEV_CUR = (r"(ON|RECEIVING|TAKING|STARTED|STARTING|CONCOMITANT|CONCURRENT|CONCURRENTLY|IN COMBINATION WITH|COMBINED WITH|TOGETHER WITH|ALONG WITH|TREATED WITH|"
           r"TREATMENT WITH|DOSE OF|INFUSIONS? OF|LAST DOSE|HELD|PLUS|INITIATED|CONTINUED|RECEIVES|RECENT)\b.{0,30}(" + BEV + r")|"
           r"(" + BEV + r").{0,40}(WAS|WERE|IS|HAD BEEN) (GIVEN|ADMINISTERED|STARTED|HELD|PAUSED|INITIATED|ADDED|CONTINUED|RESUMED|RESTARTED)|"
           r"(" + BEV + r") (INFUSION|THERAPY|TREATMENT|EVERY|Q2|Q3|\d+ ?MG)")
BEV_GENERIC = r"PRESCRIBING INFORMATION|VEGF INHIBITOR NOTED|IS KNOWN TO|ARE KNOWN TO|LABEL|RISK FACTORS? (FOR .{0,40})?(INCLUDE|SUCH AS)|CAN (IMPAIR|DELAY|INCREASE)"

def describe(text):
    full = split(text)
    ev = [s for s in full if not BOILER.search(s)]
    ev_ok = [s for s in ev if not PRE.search(s)]           # exclude pre-treatment history sentences
    r = {}
    # wound and surgical-site attributes
    r["surgical_site_v2"] = any(hit(P["site"], s) and hit(P["lesion"], s, True) and not re.search(P["placement"], s) for s in ev_ok)
    r["dehiscence"] = any(hit(P["dehiscence"], s, True) for s in ev_ok)
    r["exposure_v2"] = any(hit(P["exposure"], s, True) for s in ev_ok)
    inf_any = any(hit(P["infection_any"], s, True) for s in ev_ok)
    inf_conf = any(hit(P["infection_confirmed"], s, True) and not re.search(P["culture_negative"], s) for s in ev_ok)
    r["infection_status"] = "confirmed" if inf_conf else ("suspected" if inf_any else "none")
    r["bone_infection"] = any(hit(P["bone_infection"], s, True) for s in ev_ok)
    for k in ["iv_antibiotics", "hospitalised", "temporary_interruption", "permanent_discontinuation"]:
        r[k + "_v2"] = any(hit(P[k], s, k in ("iv_antibiotics", "hospitalised")) for s in ev_ok)
    proc_sents = [s for s in ev_ok if not PLAN.search(s) and not re.search(r"\bTUMOU?R (RESECTION|REMOVAL)|RESECTION OF (THE )?TUMOU?R|BIOPSY|RE-?RESECTION|ENHANCE|BURR HOLE|EF-?32|TRIDENT", s)]
    for k, p in PROC.items():
        r["proc_" + k] = any(hit(p, s, True) for s in proc_sents)
    r["surgery_v2"] = any(r["proc_" + k] for k in PROC)
    # falls
    for k, (p1, p2) in FALL.items():
        r["fallmech_" + k] = any(re.search(p1, s) and re.search(p2, s) and not NEG.search(s[:40]) for s in ev)
    r["fall_device_mechanism"] = r["fallmech_tripped_or_entangled_in_cable"] or r["fallmech_pulled_by_cable"] or r["fallmech_carrying_device"]
    for k, p in FALL_INJ.items():
        r["fallinj_" + k] = any(hit(p, s, True) for s in ev)
    r["array_injury_stated"] = any(re.search(ARR, s) and re.search(INJ, s) and re.search(r"CAUSED|CONTRIBUT|DUE TO|PRESSED|DUG|CUT INTO|IMPACT|(FROM|BY|ON|AGAINST) (THE |A |AN |ONE OF THE |HIS |HER )?(BROKEN |CERAMIC |EDGE OF (THE |A )?)?(TRANSDUCER )?(ARRAY|DIS[CK]|CERAMIC|CONNECTOR)", s)
                                   and not re.search(CRBO, s) and not re.search(REL_NEG, s) for s in full)
    r["injury_at_array_site"] = any(re.search(INJ, s) and re.search(r"(UNDER|BENEATH|AT THE SITE OF|WHERE) (ONE OF )?(THE )?(TRANSDUCER )?(ARRAY|DIS[CK])", s) for s in ev)
    r["array_injury_crbo"] = any(re.search(r"CONTRIBUTION OF (THE )?(TRANSDUCER )?(ARRAYS?|DEVICE|OPTUNE)", s) and re.search(CRBO, s) and re.search(INJ, s) for s in full)
    r["fall_unrelated_stated"] = any(re.search(r"FALL", s) and re.search(REL_NEG, s) for s in full)
    # attribution (full narrative)
    pos = any(hit(REL_POS, s) and not re.search(REL_NEG, s) for s in full)
    crbo = any(re.search(CRBO, s) and re.search(r"OPTUNE|DEVICE|ARRAY|TTFIELDS|NOVOTTF|CONTRIBUTION", s) for s in full)
    negr = any(re.search(REL_NEG, s) for s in full)
    r["reporter_attribution"] = "related" if pos else ("cannot_be_ruled_out" if crbo else ("not_related" if negr else "not_stated"))
    r["device_mechanism_v2"] = any(re.search(MECH, s) and not NEG.search(s[:30]) for s in ev)
    # deaths
    dsents = [s for s in full if re.search(DEATH, s) and not re.search(DEATH_GENERIC, s)]
    r["death_v2"] = any(hit(DEATH, s, True) for s in dsents)
    dev = any(re.search(D_DEV, s) for s in full)
    oth = any(re.search(D_OTHER, s) for s in full)
    r["death_attribution"] = ("not_applicable" if not r["death_v2"] else
                              "device_contribution_not_excluded" if dev else "other_cause_stated" if oth else "not_stated")
    # bevacizumab context
    bs = [s for s in full if re.search(BEV, s)]
    if not bs:
        r["bevacizumab_context"] = "not_mentioned"
    elif any(re.search(BEV_NOT, s) for s in bs):
        r["bevacizumab_context"] = "not_receiving"
    elif any(re.search(BEV_CUR, s) and not re.search(BEV_GENERIC, s) for s in bs):
        r["bevacizumab_context"] = "current_or_recent"
    elif any(re.search(BEV_PREV, s) for s in bs):
        r["bevacizumab_context"] = "previous_only"
    else:
        r["bevacizumab_context"] = "mention_only"
    return r

if __name__ == "__main__":
    d = pd.read_csv(DERIVED / "analysis_dataset.csv", usecols=["row_id", "report_number", "mdr_text_concat"])
    d["set"] = "study"
    mp = DERIVED / "missed_reports.csv"
    if mp.exists():
        m = pd.read_csv(mp, usecols=["row_id", "report_number", "mdr_text_concat"]); m["set"] = "missed"
        d = pd.concat([d, m], ignore_index=True)
    out = pd.DataFrame([describe(t) for t in d.mdr_text_concat])
    out.insert(0, "set", d["set"].values); out.insert(0, "report_number", d.report_number.values); out.insert(0, "row_id", d.row_id.values)
    out.to_csv(ANN / "attributes_rule_based_v2.csv", index=False)
    pats = {"BOILER": BOILER.pattern, "NEG": NEG.pattern, "PLAN": PLAN.pattern, "PRE": PRE.pattern, **{f"P.{k}": v for k, v in P.items()},
            **{f"PROC.{k}": v for k, v in PROC.items()}, **{f"FALL.{k}": list(v) for k, v in FALL.items()}, **{f"FALL_INJ.{k}": v for k, v in FALL_INJ.items()},
            "ARR": ARR, "DEV": DEV, "INJ": INJ, "CRBO": CRBO, "REL_POS": REL_POS, "REL_NEG": REL_NEG, "MECH": MECH, "DEATH": DEATH, "DEATH_GENERIC": DEATH_GENERIC,
            "D_OTHER": D_OTHER, "D_DEV": D_DEV, "BEV": BEV, "BEV_NOT": BEV_NOT, "BEV_PREV": BEV_PREV, "BEV_CUR": BEV_CUR, "BEV_GENERIC": BEV_GENERIC}
    json.dump(pats, open(RESULTS / "descriptor_patterns_v2.json", "w"), indent=1)
    s = out[out.set == "study"]
    print({c: (int(s[c].sum()) if s[c].dtype == bool else s[c].value_counts().to_dict()) for c in s.columns[3:]})
