"""StudyMate - AI study companion. Run: streamlit run app.py"""
import tempfile, os
import pandas as pd
import streamlit as st
import kb as K, llm
from adaptive import LearnerModel

st.set_page_config(page_title="StudyMate", page_icon="📚", layout="wide")
S = st.session_state
S.setdefault("kb", None); S.setdefault("lm", None); S.setdefault("q", None); S.setdefault("log", [])

# ---------------- sidebar: ingestion ----------------
with st.sidebar:
    st.title("📚 StudyMate")
    st.caption("LLM: " + ("Claude (online)" if llm.llm_available() else "offline fallback - set ANTHROPIC_API_KEY"))
    files = st.file_uploader("Textbooks (PDF), slides (PPTX), transcripts (VTT/SRT/TXT), audio/video",
                             accept_multiple_files=True,
                             type=["pdf", "pptx", "vtt", "srt", "txt", "mp3", "mp4", "wav", "m4a"])
    yt = st.text_input("YouTube lecture URL (captions)")
    lang = st.selectbox("Tutor language", ["English", "Hindi", "Tamil", "Bengali", "Marathi", "Telugu", "Gujarati"])
    dense = st.checkbox("Dense embeddings (sentence-transformers)", value=False)
    if st.button("Build knowledge base", type="primary"):
        recs = []
        with st.spinner("Ingesting..."):
            for f in files or []:
                ext = f.name.rsplit(".", 1)[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix="." + ext) as t:
                    t.write(f.getbuffer()); p = t.name
                try:
                    if ext == "pdf": recs += K.ingest_pdf(p, f.name)
                    elif ext == "pptx": recs += K.ingest_pptx(p, f.name)
                    elif ext in ("vtt", "srt", "txt"): recs += K.ingest_transcript(f.getvalue().decode("utf8", "ignore"), f.name)
                    else: recs += K.transcribe_media(p, f.name)
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            if yt:
                try: recs += K.ingest_youtube(yt)
                except Exception as e: st.error(f"YouTube: {e}")
            kb = K.KnowledgeBase(use_dense=dense); kb.add(recs); kb.build()
            if kb.chunks:
                S.kb, S.lm, S.q = kb, LearnerModel(list(kb.topic_names)), None
                st.success(f"{len(kb.chunks)} chunks, {len(kb.topic_names)} topics")
            else:
                st.warning("Nothing ingested.")

if not S.kb:
    st.info("Upload your lecture transcripts, textbook and slides in the sidebar, then click **Build knowledge base**.")
    st.stop()
kb, lm = S.kb, S.lm

def render_sources(chs):
    for c in chs:
        icon = {"video": "🎬", "slides": "🖼️", "textbook": "📖"}[c.kind]
        with st.expander(f"{icon} [{c.id}] {c.cite()}"):
            st.write(c.text)
            if c.link: st.markdown(f"[Jump to video moment]({c.link})")

tab_ask, tab_quiz, tab_prog = st.tabs(["💬 Ask the tutor", "🎯 Adaptive quiz", "📈 Progress & plan"])

# ---------------- ask ----------------
with tab_ask:
    q = st.text_input("Ask anything about your course")
    if q:
        hits = [c for c, _ in kb.search(q, k=5)]
        topics = {c.topic for c in hits}
        level = lm.difficulty(max(topics, key=lambda t: sum(c.topic == t for c in hits))) if hits else "medium"
        ans, used = llm.answer(q, hits, level, lang)
        st.markdown(ans)
        st.caption(f"Explanation level adapted to your mastery: **{level}**")
        render_sources(used)

# ---------------- quiz ----------------
with tab_quiz:
    c1, c2 = st.columns([3, 1])
    mode = c2.radio("Topic", ["Adaptive (weakest first)", "Choose"])
    if mode == "Choose":
        topic = c2.selectbox("Pick", list(kb.topic_names), format_func=lambda t: kb.topic_names[t])
    if c1.button("Next question") or S.q is None:
        topic = lm.next_topic() if mode.startswith("Adaptive") else topic
        d = lm.difficulty(topic)
        pool = kb.topic_chunks(topic)
        import random; ctx = random.sample(pool, min(3, len(pool)))
        qs = llm.make_questions(ctx, d, 1, lang)
        S.q = {"topic": topic, "diff": d, "ctx": ctx, "item": qs[0] if qs else None, "done": False}
    cur = S.q
    if cur and cur["item"]:
        it = cur["item"]
        st.markdown(f"**Topic:** {kb.topic_names[cur['topic']]} · **Difficulty:** {cur['diff']}")
        st.subheader(it["question"])
        choice = st.radio("Answer", range(4), format_func=lambda i: it["options"][i], index=None, key=f"r{id(it)}")
        if choice is not None and not cur["done"] and st.button("Submit"):
            ok = choice == it["answer"]; cur["done"] = True
            lm.update(cur["topic"], ok, it["source"])
            S.log.append({"topic": kb.topic_names[cur["topic"]], "correct": ok, "difficulty": cur["diff"]})
            (st.success if ok else st.error)("Correct!" if ok else "Not quite.")
            if not ok: st.markdown(llm.explain_mistake(it, choice, cur["ctx"]))
            else: st.caption(it["explanation"])
            render_sources([c for c in cur["ctx"] if c.id == it["source"]])
    elif cur:
        st.warning("Couldn't generate a question for this topic; click Next.")

# ---------------- progress ----------------
with tab_prog:
    rows = lm.study_plan(kb.topic_names)
    df = pd.DataFrame(rows, columns=["Topic", "Mastery", "Attempts", "Recently missed sources"])
    st.bar_chart(df.set_index("Topic")["Mastery"])
    st.dataframe(df, use_container_width=True)
    weak = [r for r in rows if r[1] < 0.6][:3]
    st.subheader("Suggested study plan")
    if not S.log: st.write("Take a few quiz questions to personalise your plan.")
    for name, p, n, missed in weak:
        st.write(f"- **{name}** (mastery {p:.0%}): review " + (", ".join(missed) if missed else "the topic's sources") + " then retry.")
