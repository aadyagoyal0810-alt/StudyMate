# VORTEX 2K26: Project Submission Pack

**Theme:** Future of Education & Employability (also touches AI for Bharat & Public Services: accessibility, multilingual communication)
**Project name:** StudyMate: a Source-Cited, Multilingual AI Study Companion
**Tagline:** Turn any lecture video, textbook and slide deck into a personal tutor that cites its sources, speaks your language and quizzes you where you are weakest.

StudyMate merges lecture videos, textbooks and slides into one knowledge base where every answer cites its exact source: page, slide or video timestamp. It tutors in English and Indian languages, generates quiz questions from the student's own material, and uses Bayesian knowledge tracing to focus practice on weak topics, with a study plan pointing to what to revisit.

## Problem statement
Students in India and elsewhere study from disconnected resources: recorded lectures, textbooks and slides, mostly in English. Revision means hunting across all of them. General chatbots hallucinate and give no way to verify answers, and generic quizzes do not adapt to what a learner already knows. Students who are more comfortable in their mother tongue are left behind.

## Solution
1. **Unified knowledge base:** ingests PDFs, PPTX, lecture transcripts (VTT/SRT/TXT), YouTube captions and raw audio/video (Whisper). Every chunk keeps its source and location, and video moments get deep links.
2. **Verifiable tutoring:** answers come only from the student's materials with `[S#]` citations. A validator strips citations that were not retrieved, and the tutor refuses when the material doesn't cover a question.
3. **Multilingual:** the tutor and quizzes work in Hindi, Tamil, Bengali, Marathi, Telugu and Gujarati, while citations stay linked to the original sources.
4. **Adaptive assessment:** questions are generated from the sources, and each cites the passage that proves the answer. Bayesian knowledge tracing picks the weakest topic and sets the difficulty.
5. **Personalised study plan:** shows mastery per topic and lists the exact sources behind recent mistakes.

## Impact
- **Students:** fewer hours spent searching, and trustworthy answers they can verify in one click.
- **Accessibility:** learning in the student's own language.
- **Employability:** the same pipeline works on certification and interview-prep material, so learners can upskill efficiently.
- **Scalable:** runs on a laptop with only TF-IDF, with optional dense embeddings and a hosted LLM.

## Tech stack
Python, Streamlit, scikit-learn (TF-IDF, KMeans), optional sentence-transformers, pypdf, python-pptx, faster-whisper, youtube-transcript-api, Anthropic Claude API (generation, grounded answering, question writing).

## Innovation highlights
Multi-modal unification with timestamp deep links; citation validation on answers and quiz items; a principled learner model (BKT) rather than a simple "harder if correct" rule; multilingual output with source-faithful citations; offline fallback.

## Future scope
Spaced repetition, topic prerequisite graph, figure and diagram understanding from slides, voice tutoring, teacher dashboard for class-level weak spots, WhatsApp bot for low-bandwidth access.
