# Music App Backend

This is the Node.js backend for the lesson transcription app.

## Render settings

```text
Root Directory: backend
Build Command: npm install && npm run build
Start Command: npm run start
```

## Required environment variables

```text
OPENAI_API_KEY=your OpenAI API key
HOST=0.0.0.0
OPENAI_TRANSCRIPTION_MODEL=gpt-4o-transcribe-diarize
OPENAI_TRANSCRIPTION_LANGUAGE=ja
FORCE_JAPANESE_TRANSCRIPTS=true
OPENAI_TRANSCRIPT_NORMALIZER_MODEL=gpt-4.1-mini
```

After deploy, open `/status` on the Render URL.
