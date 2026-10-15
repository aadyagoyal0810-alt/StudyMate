"""LLM layer. Uses Anthropic API if ANTHROPIC_API_KEY is set; otherwise a grounded offline fallback
(extractive answers + cloze questions) so the app always runs for demo/testing."""
import os, json, re, random

MODEL = os.getenv("STUDYMATE_MODEL", "claude-sonnet-5-5")


def _client():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key: return None
    import anthropic
    return anthropic.Anthropic(api_key=key)


def _ask(system, user, max_tokens=1200):
    c = _client()
    r = c.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                          messages=[{"role": "user", "content": user}])
    return r.content[0].text


def llm_available(): return bool(os.getenv("ANTHROPIC_API_KEY"))


def _ctx(chunks):
    return "\n\n".join(f"[{c.id}] ({c.kind}: {c.cite()})\n{c.text}" for c in chunks)


def _validate_cites(text, chunks):
    valid = {c.id for c in chunks}
    return text, sorted(set(re.findall(r"\[(S\d+)\]", text)) - valid)


TUTOR_SYS = """You are StudyMate, a patient tutor. Answer ONLY from the numbered sources.
Cite every factual sentence with its source id in square brackets, e.g. [S3]. If the sources do not
contain the answer, say so plainly instead of guessing. Adapt depth to the learner level given.
End with one short check-for-understanding question."""


def answer(question, chunks, level="medium", language="English"):
    if not chunks:
        return "I couldn't find this in your course materials, so I won't guess. Try rephrasing or upload more sources.", []
    if llm_available():
        txt = _ask(TUTOR_SYS, f"Learner level: {level}\nReply in: {language} (keep [S#] citations unchanged)\n\nSOURCES:\n{_ctx(chunks)}\n\nQUESTION: {question}")
        txt, bad = _validate_cites(txt, chunks)
        if bad: txt += f"\n\n_(Warning: removed unverifiable citations {bad})_"; 
        for b in bad: txt = txt.replace(f"[{b}]", "")
        return txt, chunks
    top = chunks[0]
    sent = re.split(r"(?<=[.!?])\s+", top.text)
    return " ".join(sent[:3]) + f" [{top.id}]\n\n_(offline mode: extractive answer. Set ANTHROPIC_API_KEY for full tutoring.)_", chunks


QUIZ_SYS = """Write {n} multiple-choice questions strictly from the numbered sources, difficulty: {d}.
easy=recall, medium=understanding/application, hard=multi-step reasoning or comparing sources.
Return ONLY a JSON list. Each item: {{"question":str,"options":[4 strings],"answer":int(0-3),
"explanation":str,"source":"S<id>" (the single source that proves the answer)}}.
Distractors must be plausible. Never use facts absent from the sources."""


def make_questions(chunks, difficulty="medium", n=1, language="English"):
    if llm_available():
        raw = _ask(QUIZ_SYS.format(n=n, d=difficulty) + f"\nWrite questions, options and explanations in {language}.", _ctx(chunks), 1500)
        try:
            items = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
            valid = {c.id for c in chunks}
            return [q for q in items if q.get("source") in valid and len(q.get("options", [])) == 4][:n]
        except Exception:
            pass
    return _cloze(chunks, n)


def _cloze(chunks, n):
    """Offline fallback: blank out a key term; distractors drawn from other chunks in the topic."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    pool = [w for c in chunks for w in set(re.findall(r"[A-Za-z]{6,}", c.text))]
    out = []
    for c in random.sample(chunks, min(n, len(chunks))):
        sents = [s for s in re.split(r"(?<=[.!?])\s+", c.text) if 8 <= len(s.split()) <= 40]
        if not sents: continue
        s = random.choice(sents)
        words = re.findall(r"[A-Za-z]{6,}", s)
        if not words: continue
        ans = max(words, key=len)
        distract = random.sample([w for w in pool if w.lower() != ans.lower()], k=min(3, len(pool) - 1))
        opts = distract + [ans]; random.shuffle(opts)
        out.append({"question": "Fill in the blank: " + s.replace(ans, "_____", 1), "options": opts,
                    "answer": opts.index(ans), "explanation": f"The source says: \"{s}\"", "source": c.id})
    return out


def explain_mistake(q, chosen, chunks):
    src = next((c for c in chunks if c.id == q["source"]), None)
    if llm_available() and src:
        return _ask("You are a tutor. In <=80 words, explain why the learner's choice is wrong and the "
                    "correct answer right. Cite the source id.",
                    f"Question: {q['question']}\nLearner chose: {q['options'][chosen]}\n"
                    f"Correct: {q['options'][q['answer']]}\nSOURCE [{src.id}]: {src.text}")
    return q["explanation"] + (f" [{q['source']}]" if q.get("source") else "")
