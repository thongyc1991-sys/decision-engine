import json
import threading
import uuid
from datetime import datetime, timezone

import streamlit as st

st.set_page_config(
    page_title="Decision Intelligence Engine",
    page_icon="🧭",
    layout="centered",
)

MODEL = "google/flan-t5-small"
VERSION = "online-demo-1"

QUESTIONS = {
    "decision": "What decision are you facing?",
    "context": "What is your education, career or business situation?",
    "options": "What options are you considering?",
    "goals": "What outcome would you like within the next year?",
    "values": "What matters most to you when choosing?",
    "strengths": "What strengths do you have? Give an example.",
    "constraints": "What time, money or family constraints matter?",
    "evidence": "What have you tried, and what did you learn?",
    "uncertainty": "What worries you or remains unknown?",
    "next_step": "What small, reversible step could you try?",
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


@st.cache_resource
def load_ai():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL,
        use_safetensors=True,
    )
    model.eval()
    return tokenizer, model, threading.Lock()


def ai_suggestion(answers):
    import torch

    tokenizer, model, lock = load_ai()

    # Each answer is short so the small model can read the whole intake.
    context = "\n".join(
        f"{key}: {value}" for key, value in answers.items()
    )

    prompt = (
        "Suggest one cautious question a career or business coach "
        "could discuss with this person. Do not diagnose them or "
        "claim that one option is definitely correct.\n"
        + context
        + "\nDiscussion question:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    # Avoid silently discarding questionnaire answers.
    if inputs["input_ids"].shape[1] > 512:
        raise ValueError(
            "The questionnaire is too long for this small model. "
            "Create another assessment using shorter answers."
        )

    with lock, torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    return tokenizer.decode(
        output[0], skip_special_tokens=True
    ).strip()


def validate_backup(data):
    if not isinstance(data, dict):
        raise ValueError("Invalid backup.")

    if data.get("format") != VERSION:
        raise ValueError("This is not a compatible backup.")

    records = data.get("records")

    if not isinstance(records, dict) or len(records) > 100:
        raise ValueError("Invalid or oversized record collection.")

    for key, record in records.items():
        if not isinstance(key, str) or not isinstance(record, dict):
            raise ValueError("Invalid assessment.")

        if not isinstance(record.get("code"), str):
            raise ValueError("Missing client code.")

        if not isinstance(record.get("created"), str):
            raise ValueError("Missing creation date.")

        answers = record.get("answers", {})
        if not isinstance(answers, dict):
            raise ValueError("Invalid answers.")

        if set(answers) != set(QUESTIONS):
            raise ValueError("Questionnaire does not match this version.")

        if not all(
            isinstance(v, str) and len(v) <= 140
            for v in answers.values()
        ):
            raise ValueError("Invalid questionnaire answer.")

        for collection in ("reports", "reviews"):
            items = record.get(collection)
            if not isinstance(items, list):
                raise ValueError("Invalid report or review history.")
            if not all(isinstance(item, dict) for item in items):
                raise ValueError("Invalid history entry.")

        for report in record["reports"]:
            if not all(
                isinstance(report.get(field), str)
                for field in ("id", "created", "model", "suggestion")
            ):
                raise ValueError("Invalid report.")

        for review in record["reviews"]:
            if not all(
                isinstance(review.get(field), str)
                for field in (
                    "report_id", "created", "reviewer", "corrections",
                    "assessment", "actions", "outcome",
                )
            ):
                raise ValueError("Invalid review.")

    return records


if "records" not in st.session_state:
    st.session_state.records = {}

records = st.session_state.records

st.title("🧭 Decision Intelligence Engine")
st.caption("Free online demonstration • Questionnaire → AI → Human review")

st.warning(
    "Use fictional or anonymised information. This is not a validated "
    "assessment tool. AI suggestions can be inaccurate. "
    "Download your records before leaving: session data can disappear."
)

with st.expander("Restore a previously downloaded backup"):
    uploaded = st.file_uploader(
        "Choose your JSON backup",
        type=["json"],
    )

    if st.button("Restore backup"):
        try:
            if uploaded is None:
                raise ValueError("Choose a backup first.")

            if uploaded.size > 2_000_000:
                raise ValueError("Backup must be smaller than 2 MB.")

            restored = validate_backup(json.loads(uploaded.getvalue()))

            # Do not silently overwrite current records.
            if set(restored) & set(records):
                raise ValueError(
                    "Some assessments already exist in this session. "
                    "Start a fresh session to restore this backup."
                )

            records.update(restored)
            st.success("Backup restored.")
        except (ValueError, TypeError, KeyError):
            st.error(
                "Could not restore this backup. Check its format, "
                "size and whether those assessments already exist."
            )

st.subheader("1. Questionnaire")

with st.form("questionnaire", clear_on_submit=True):
    code = st.text_input(
        "Client code — avoid real names",
        max_chars=60,
        placeholder="Example: TEST-001",
    )

    st.caption(
        "Keep each answer brief: maximum 140 characters. "
        "This limit suits the small free demonstration model."
    )

    answers = {
        key: st.text_area(label, max_chars=140, height=85)
        for key, label in QUESTIONS.items()
    }

    consent = st.checkbox(
        "The participant agrees to online processing and human review."
    )

    submitted = st.form_submit_button("Save questionnaire")

    if submitted:
        if not code.strip():
            st.error("Enter a client code.")
        elif not all(value.strip() for value in answers.values()):
            st.error("Please answer all questions.")
        elif not consent:
            st.error("Agreement is required before saving.")
        else:
            case_id = uuid.uuid4().hex
            records[case_id] = {
                "code": code.strip(),
                "created": timestamp(),
                "version": VERSION,
                "consent": True,
                "answers": {
                    key: value.strip()
                    for key, value in answers.items()
                },
                "reports": [],
                "reviews": [],
            }
            st.success("Questionnaire saved in this session.")

if records:
    st.subheader("2. Preliminary report")

    case_id = st.selectbox(
        "Choose an assessment",
        options=list(records),
        format_func=lambda key: (
            f"{records[key]['code']} · {key[:8]}"
        ),
    )

    case = records[case_id]

    with st.expander("View questionnaire answers"):
        for key, label in QUESTIONS.items():
            st.write(label)
            st.text(case["answers"][key])

    if st.button("Generate AI discussion point"):
        try:
            with st.spinner(
                "Loading the free model and generating a suggestion…"
            ):
                suggestion = ai_suggestion(case["answers"])

            if not suggestion:
                raise ValueError("The model returned no suggestion.")

            case["reports"].append({
                "id": uuid.uuid4().hex,
                "created": timestamp(),
                "model": MODEL,
                "suggestion": suggestion,
            })

            st.success("A new preliminary report version was saved.")

        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error(
                "The AI model could not load or run. "
                "Your questionnaire remains in this session. "
                "Try again later and download a backup."
            )

    if case["reports"]:
        report_index = st.selectbox(
            "Report version",
            options=list(range(len(case["reports"]))),
            index=len(case["reports"]) - 1,
            format_func=lambda index: (
                f"Version {index + 1} · "
                f"{case['reports'][index]['created']}"
            ),
        )

        report = case["reports"][report_index]

        st.info("PRELIMINARY — HUMAN REVIEW REQUIRED")

        st.write("Participant's stated decision")
        st.text(case["answers"]["decision"])

        st.write("Participant's stated options")
        st.text(case["answers"]["options"])

        st.write("Participant's stated goal")
        st.text(case["answers"]["goals"])

        st.write("AI-generated discussion point")
        st.text(report["suggestion"])

        st.write("Standard questions for the face-to-face session")
        st.write(
            "• What evidence supports each option?\n"
            "• What trade-offs are acceptable?\n"
            "• What important information is still missing?\n"
            "• What small experiment would reduce uncertainty?"
        )

        st.caption(
            "The participant summary and standard questions above are "
            "templates. Only the labelled discussion point is AI-generated."
        )

        st.subheader("3. Human review and feedback")

        with st.form("review_" + report["id"], clear_on_submit=True):
            reviewer = st.text_input(
                "Reviewer code", max_chars=60
            )

            accuracy = st.slider(
                "How useful was the AI discussion point?",
                min_value=1,
                max_value=5,
                value=3,
            )

            corrections = st.text_area(
                "Incorrect claims or missing information",
                max_chars=3000,
            )
            assessment = st.text_area(
                "Your assessment after meeting the participant",
                max_chars=3000,
            )
            actions = st.text_area(
                "Agreed actions and follow-up date",
                max_chars=3000,
            )
            outcome = st.text_area(
                "Observed outcome — enter Pending if not yet known",
                max_chars=3000,
            )

            if st.form_submit_button("Save human review"):
                values = [
                    reviewer, corrections, assessment, actions, outcome
                ]

                if not all(value.strip() for value in values):
                    st.error(
                        "Complete each field. Use 'None identified' "
                        "or 'Pending' where appropriate."
                    )
                else:
                    case["reviews"].append({
                        "report_id": report["id"],
                        "created": timestamp(),
                        "reviewer": reviewer.strip(),
                        "usefulness": accuracy,
                        "corrections": corrections.strip(),
                        "assessment": assessment.strip(),
                        "actions": actions.strip(),
                        "outcome": outcome.strip(),
                    })
                    st.success("Human review saved in this session.")

    if case["reviews"]:
        with st.expander("View human review history"):
            for review in case["reviews"]:
                st.json(review)

    st.subheader("4. Keep your records")

    st.download_button(
        "Download all questionnaires, reports and feedback",
        data=json.dumps(
            {"format": VERSION, "records": records},
            indent=2,
        ),
        file_name="decision-engine-backup.json",
        mime="application/json",
    )

    st.caption(
        "Download again after every change. Upload this file using "
        "'Restore a previously downloaded backup' on your next visit."
    )

    confirm_delete = st.checkbox(
        "I want to remove the selected assessment from this session."
    )

    if st.button("Delete selected assessment", disabled=not confirm_delete):
        del records[case_id]
        st.rerun()

st.divider()
st.caption(
    "Feedback is recorded for human comparison. It does not automatically "
    "train the AI. This demonstration has no client accounts, permanent "
    "database, formal consent records or clinical validation."
)
