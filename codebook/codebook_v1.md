**MAUDE TTFields Adverse Event  
Classification Codebook**

Version 1.0 — April 2026

# 1. Purpose and Scope

This codebook defines the standardised classification system for adverse events associated with Tumor Treating Fields (TTFields / Optune) devices as reported to the FDA MAUDE database. It is the authoritative reference for all three classification arms: manual human classification (medical student), LLM-based classification (Qwen 2.5, Llama 3.1 via Ollama), and the rule-based regex pipeline.

All classifiers must assign values on the same five dimensions using the same codes defined below. This ensures comparability across arms and enables valid inter-rater agreement analysis.

### Input Data

The primary input for classification is the mdr_text_concat field, which contains the concatenated free-text narrative from each MAUDE report. Supplementary structured fields (event_type, brand_name, model_number) may be used for context but should not override the narrative-based classification.

# 2. Dimension 1: Primary Adverse Event Category

Mandatory. Single-select. Choose exactly ONE category for each report.

## SKIN_NONINF: Non-Infectious Skin Reactions

**Definition:** Adverse skin reactions at array or electrode sites that do NOT involve infection.

**Inclusion criteria:**

• Dermatitis or eczematous reactions (contact, irritant, or allergic)

• Erythema (redness) at array site

• Rash or maculopapular eruptions

• Ulceration or blistering under arrays

• Wound dehiscence (separation) WITHOUT infection

• Skin erosion, peeling, or breakdown

• Scalp burns or thermal skin injury from arrays

• Skin soreness or irritation requiring array relocation

**Exclusion criteria:**

• Any lesion with confirmed or suspected infection → INFECTION

• Skin damage caused by device overheating → HARDWARE (primary), SKIN_NONINF (secondary)

• Post-surgical wound complications not related to arrays

**Example:** *"SUBJECT EXPERIENCED SKIN IRRITATION/RASH UNDER THE TRANSDUCER ARRAYS. TREATMENT WAS TEMPORARILY PAUSED TO ALLOW HEALING. ARRAYS WERE RELOCATED AND TREATMENT RESUMED."*

## INFECTION: Infection

**Definition:** Any infection, whether localised to the device site or systemic, that is reported in connection with TTFields use.

**Inclusion criteria:**

• Wound infection at array site

• Cellulitis at or near array placement

• Abscess formation

• Osteomyelitis of the skull

• Meningitis

• Sepsis or systemic infection

• Infected skin breakdown or wound dehiscence WITH infection

• MRSA, Staphylococcal, or other named organism infection

• Reports mentioning IV antibiotic treatment for device-site infection

**Exclusion criteria:**

• Sterile inflammation without evidence of infection → SKIN_NONINF

• Infection unrelated to device site (e.g., UTI, pneumonia without device connection) → SYSTEMIC

• Mention of prophylactic antibiotics without actual infection

**Example:** *"PATIENT DEVELOPED WOUND INFECTION AT ARRAY SITE REQUIRING IV ANTIBIOTICS AND HOSPITALIZATION. OPTUNE TREATMENT DISCONTINUED."*

## HARDWARE: Device Malfunction / Hardware Failure

**Definition:** Malfunction or failure of any device component not attributable to user error.

**Inclusion criteria:**

• Battery failure or inability to charge

• Transducer array defect or damage

• Cable or connector damage, fraying, or breakage

• Power supply or adapter malfunction

• Device overheating or thermal shutdown

• System error codes or display errors

• Device ceasing to function or becoming inoperable

• Electrical fault or short circuit

• Firmware or software malfunction

**Exclusion criteria:**

• Damage caused by patient misuse → USER_ERROR

• Skin reaction from a functioning array → SKIN_NONINF

**Example:** *"NOVOCURE WAS NOTIFIED THAT THE DEVICE BATTERY STOPPED CHARGING. THE PATIENT RETURNED THE DEVICE. INVESTIGATION REVEALED A FAULTY BATTERY CONNECTION."*

## FALL_ENTANGLE: Falls and Entanglement

**Definition:** Mobility-related incidents caused or facilitated by the device or its cables.

**Inclusion criteria:**

• Falls due to tripping over cables or cords

• Patient pulled backward/down by cord reaching maximum length

• Entanglement or strangulation risk from cables

• Fractures or injuries resulting from device-related falls

• Near-miss incidents involving cables and mobility

**Exclusion criteria:**

• Falls clearly caused by seizure → NEURO

• Falls due to disease-related weakness without device involvement → DISEASE_PROG

• Falls in context of generalised debility without cable/device involvement

**Example:** *"SUBJECT WAS WEARING THE NOVO-TTF-100A DEVICE PLUGGED INTO AN ELECTRIC SOURCE ON THE WALL WHEN THE SUBJECT WALKED AWAY WHILE STILL CONNECTED. THE CORD REACHED ITS MAXIMUM LENGTH AND PULLED THE SUBJECT BACKWARD, RESULTING IN A FALL."*

## NEURO: Neurological Events

**Definition:** Neurological symptoms or events reported during TTFields therapy. Note: causality is often uncertain due to underlying GBM.

**Inclusion criteria:**

• Seizures (new-onset or worsening)

• Persistent or worsening headaches

• Cognitive decline or dysfunction

• Confusion or disorientation

• Focal neurological deficits (weakness, numbness, aphasia)

• Altered mental status

• Status epilepticus

**Exclusion criteria:**

• Neurological deterioration clearly attributed to tumour progression in the narrative → DISEASE_PROG

• Seizure causing a fall → NEURO (primary); note fall as secondary if significant

**Example:** *"PATIENT EXPERIENCED NEW-ONSET SEIZURE ACTIVITY WHILE WEARING THE OPTUNE DEVICE. ANTI-EPILEPTIC MEDICATION WAS ADJUSTED. RELATIONSHIP TO DEVICE USE VERSUS DISEASE PROGRESSION IS UNCERTAIN."*

## SYSTEMIC: Systemic Complications

**Definition:** Organ-level or systemic complications occurring during TTFields therapy.

**Inclusion criteria:**

• Pulmonary embolism (PE)

• Deep vein thrombosis (DVT)

• Cerebrospinal fluid (CSF) leakage

• Intracranial haemorrhage

• Pneumonia or respiratory failure

• Cardiac arrest or myocardial events

• Renal failure

• Other thromboembolic events

**Exclusion criteria:**

• Systemic infection → INFECTION

• Neurological events → NEURO

• Disease progression → DISEASE_PROG

**Example:** *"PATIENT TREATED WITH OPTUNE DEVELOPED PULMONARY EMBOLISM REQUIRING HOSPITALISATION AND ANTICOAGULATION THERAPY. OPTUNE USE WAS SUSPENDED."*

## USER_ERROR: User/Application Error

**Definition:** Adverse events resulting from incorrect use of the device by the patient, caregiver, or healthcare professional.

**Inclusion criteria:**

• Incorrect array placement

• Wrong device settings

• Poor treatment adherence or non-compliance

• Improper cable handling causing damage

• Failure to follow array rotation schedule

• Patient operating device against medical advice

**Exclusion criteria:**

• Device malfunction not caused by user → HARDWARE

• Non-compliance leading to disease progression → DISEASE_PROG

**Example:** *"PATIENT PLACED ARRAYS IN INCORRECT POSITION, RESULTING IN SUBOPTIMAL FIELD DELIVERY. CLINICIAN IDENTIFIED THE ERROR DURING FOLLOW-UP AND PROVIDED RETRAINING."*

## DISEASE_PROG: Disease Progression

**Definition:** Reports primarily reflecting underlying disease progression rather than a device-specific adverse event.

**Inclusion criteria:**

• Tumour growth or recurrence on imaging

• Clinical deterioration attributed to GBM in the narrative

• Death from underlying cancer

• Reports filed as routine safety updates documenting disease course

• Treatment discontinuation due to disease progression (not device issues)

**Exclusion criteria:**

• Disease progression with a concurrent device-related event → classify the device event as primary

• Death of uncertain cause → evaluate other categories first; use DISEASE_PROG only if narrative indicates disease as cause

**Example:** *"PATIENT WITH RECURRENT GBM BEING TREATED WITH OPTUNE PRESENTED WITH DISEASE PROGRESSION. MRI SHOWED INCREASED TUMOR SIZE. OPTUNE TREATMENT WAS DISCONTINUED."*

## OTHER: Other

**Definition:** Events that do not fit any of the above categories but contain sufficient information for characterisation.

**Inclusion criteria:**

• Psychological events (e.g., suicidal ideation) without fitting NEURO criteria

• Allergic reactions to non-array components

• Events involving ancillary equipment (shaving, adhesives)

• Social or quality-of-life impacts reported as adverse events

**Exclusion criteria:**

• Events that fit any specific category above should be classified there, not here

**Example:** *"PATIENT REPORTED SIGNIFICANT DISTRESS AND SOCIAL ISOLATION DUE TO VISIBLE DEVICE WEARING. NO PHYSICAL ADVERSE EVENT OCCURRED."*

## UNCLASSIFIABLE: Unclassifiable

**Definition:** Reports with insufficient information to assign any meaningful adverse event category.

**Inclusion criteria:**

• Narratives too brief or vague to determine event type

• Heavily redacted reports with no clinical information

• Reports describing only administrative actions without clinical events

• Narratives referencing events but providing no details

**Exclusion criteria:**

• Reports that are lengthy but ambiguous should still be classified if ANY category can be reasonably inferred

**Example:** *"A REPORT WAS RECEIVED REGARDING AN EVENT ASSOCIATED WITH THE OPTUNE DEVICE. NO FURTHER DETAILS ARE AVAILABLE AT THIS TIME."*

# 3. Dimension 2: Severity

Mandatory. Single-select.

| **Code** | **Definition**                                     | **Indicators**                                                                                                              |
|----------|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| MILD     | Self-limiting; no medical intervention required    | Skin redness resolved spontaneously; minor discomfort; no treatment change                                                  |
| MODERATE | Outpatient medical treatment required              | Topical medications prescribed; oral antibiotics; temporary treatment pause; clinic visit                                   |
| SEVERE   | Hospitalisation, surgery, or significant morbidity | Admitted to hospital; surgical debridement; ICU stay; treatment permanently discontinued; significant functional impairment |
| FATAL    | Patient death reported                             | Any report where the patient died, regardless of causal attribution to the device                                           |

# 4. Dimension 3: Causality

Mandatory. Single-select. Assess the relationship between the device and the adverse event.

| **Code**     | **Definition**              | **Guidance**                                                                                                                                                                                   |
|--------------|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DEFINITE     | Clear device causation      | Event at array site with direct mechanistic link (e.g., thermal burn from overheating array); fall caused by cable; documented device malfunction causing the event                            |
| PROBABLE     | Likely device-related       | Strong temporal and spatial relationship (e.g., skin reaction at array site); event occurred during active device use with plausible mechanism; alternative explanations not entirely excluded |
| POSSIBLE     | Uncertain relationship      | Event occurred during TTFields therapy but could equally be disease-related or coincidental. Default for neurological events in GBM patients unless additional evidence is present             |
| UNLIKELY     | Probably not device-related | Narrative suggests the event is due to disease progression, concurrent therapy, or unrelated medical condition                                                                                 |
| UNASSESSABLE | Cannot determine            | Insufficient detail in narrative to make any causality judgment; heavily redacted reports; vague event descriptions                                                                            |

# 5. Dimension 4: Device Component

Mandatory. Multi-select (separate with \| when multiple apply). Identify which device component(s) are mentioned or implicated in the adverse event.

| **Code**    | **Description**                                               |
|-------------|---------------------------------------------------------------|
| ARRAY       | Transducer arrays (electrode pads adhered to scalp)           |
| CABLE       | Connection cables/cords between arrays and device unit        |
| BATTERY     | Battery pack, charger, power supply, or adapter               |
| UNIT        | Main device unit, field generator, or controller box          |
| SOFTWARE    | Device firmware, software, or digital interface               |
| NONE        | No device component involved (e.g., disease progression)      |
| UNSPECIFIED | A component is likely involved but not named in the narrative |

# 6. Dimension 5: Secondary Event Category

Optional. Single-select. Uses the same codes as Dimension 1. Assigned only when a report describes two clinically distinct adverse events. Leave empty if only one event type is present.  
  
The secondary category captures the second most clinically significant event. Example: device overheating (HARDWARE) causing a scalp burn (SKIN_NONINF) → primary = HARDWARE, secondary = SKIN_NONINF.

# 7. Decision Rules for Ambiguous Cases

**Rule 1: Skin + Infection co-occurrence.** If a skin lesion is described AND there is evidence of infection (pus, named organism, antibiotics for infection), classify as INFECTION primary, SKIN_NONINF secondary. If skin lesion is described WITHOUT infection evidence, classify as SKIN_NONINF only.

**Rule 2: Multiple events of different severity.** Assign primary to the more clinically severe event. Example: mild skin rash + hospitalisation for PE → primary = SYSTEMIC.

**Rule 3: Device malfunction causing injury.** If the root cause is a hardware failure that caused a secondary clinical event, the primary is HARDWARE. Example: overheating → burn = HARDWARE primary, SKIN_NONINF secondary.

**Rule 4: Neurological events in GBM patients.** Default causality = POSSIBLE unless the narrative provides specific evidence linking the event to the device (then PROBABLE/DEFINITE) or explicitly attributes it to disease (then UNLIKELY).

**Rule 5: Death reports.** Always assign severity = FATAL. For primary category: if a specific mechanism is described (e.g., PE), use that category (SYSTEMIC). If death is attributed to disease, use DISEASE_PROG. If no mechanism is clear, use UNCLASSIFIABLE.

**Rule 6: Reports describing only treatment discontinuation.** If the discontinuation is due to a specific adverse event, classify that event. If discontinuation is due to disease progression or patient choice without an adverse event, use DISEASE_PROG or OTHER.

**Rule 7: Negated events.** If the narrative states that an event did NOT occur (e.g., "no infection was observed"), do NOT classify that event. Only classify events that are affirmed as having occurred.

**Rule 8: Priority when exactly tied.** If two events are genuinely equal in clinical significance: choose the event more directly attributable to the device as primary.

# 8. Output Format (LLM JSON Schema)

LLM classifiers must return a JSON object with the following structure. Human and regex classifiers populate equivalent columns in the Excel workbook.

{  
"primary_category": "SKIN_NONINF",  
"severity": "MODERATE",  
"causality": "PROBABLE",  
"device_component": "ARRAY",  
"secondary_category": "",  
"reasoning": "Patient developed dermatitis under transducer arrays requiring topical treatment."  
}

**Valid values:** primary_category and secondary_category: SKIN_NONINF \| INFECTION \| HARDWARE \| FALL_ENTANGLE \| NEURO \| SYSTEMIC \| USER_ERROR \| DISEASE_PROG \| OTHER \| UNCLASSIFIABLE. severity: MILD \| MODERATE \| SEVERE \| FATAL. causality: DEFINITE \| PROBABLE \| POSSIBLE \| UNLIKELY \| UNASSESSABLE. device_component: ARRAY \| CABLE \| BATTERY \| UNIT \| SOFTWARE \| NONE \| UNSPECIFIED (pipe-separated for multi-select). reasoning: Free-text justification (1 sentence).
