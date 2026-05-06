import cors from "cors";
import dotenv from "dotenv";
import express, { type NextFunction, type Request, type Response } from "express";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { createReadStream } from "node:fs";
import http from "node:http";
import path from "node:path";
import OpenAI from "openai";
import { WebSocketServer, type RawData, type WebSocket } from "ws";

// Load environment variables from .env before reading process.env.
dotenv.config();

const app = express();

const port = Number(process.env.PORT ?? 3000);
const host = process.env.HOST ?? "0.0.0.0";
const corsOrigin = process.env.CORS_ORIGIN ?? "*";
const uploadDir = path.resolve(process.env.UPLOAD_DIR ?? "uploads");
const audioUploadDir = path.join(uploadDir, "audio");
const transcriptUploadDir = path.join(uploadDir, "transcripts");
const transcriptionModel =
  process.env.OPENAI_TRANSCRIPTION_MODEL ?? "gpt-4o-transcribe-diarize";
const transcriptionLanguage = process.env.OPENAI_TRANSCRIPTION_LANGUAGE ?? "ja";
const musicTerms =
  process.env.MUSIC_TERMS ??
  [
    "ドレミファソラシド",
    "半音",
    "全音",
    "スケール",
    "メジャースケール",
    "マイナースケール",
    "クロマチックスケール",
    "コード",
    "コードトーン",
    "キー",
    "テンポ",
    "リズム",
    "拍",
    "拍子",
    "裏拍",
    "音程",
    "ピッチ",
    "チューニング",
    "運指",
    "替え指",
    "タンギング",
    "シングルタンギング",
    "ダブルタンギング",
    "スラー",
    "スタッカート",
    "アクセント",
    "レガート",
    "ロングトーン",
    "ビブラート",
    "ブレス",
    "息",
    "息のスピード",
    "アンブシュア",
    "マウスピース",
    "リード",
    "リガチャー",
    "ネック",
    "ベル",
    "オクターブキー",
    "サブトーン",
    "オーバートーン",
    "フラジオ",
    "アーティキュレーション",
    "ニュアンス",
    "フレーズ",
    "アドリブ",
    "ソロ",
    "テーマ",
    "メロディ",
    "ハーモニー",
    "ジャズ",
    "ブルース",
    "ツーファイブ",
    "ツーファイブワン",
    "循環",
    "II-V-I",
    "ブルーノート",
  ].join("、");
const transcriptionPrompt =
  process.env.OPENAI_TRANSCRIPTION_PROMPT ??
  `これは日本語のサックスレッスン音声です。英語として聞こえる断片も、原則として日本語の音楽レッスン文脈で解釈してください。特に次の音楽用語を優先して認識してください: ${musicTerms}。「あー」「えっと」「うーん」「あの」「その」などの意味の薄いフィラーや言いよどみは省き、話の内容が分かる自然な日本語の文章として文字起こししてください。`;
const transcriptionUsesDiarization = transcriptionModel.includes("diarize");
const liveTranscriptionChunkSeconds = Number(
  process.env.LIVE_TRANSCRIPTION_CHUNK_SECONDS ?? 3,
);
const forceJapaneseTranscripts =
  process.env.FORCE_JAPANESE_TRANSCRIPTS !== "false";
const transcriptNormalizerModel =
  process.env.OPENAI_TRANSCRIPT_NORMALIZER_MODEL ?? "gpt-4.1-mini";
const speaker1Label =
  process.env.SPEAKER_1_LABEL ?? process.env.TEACHER_SPEAKER_LABEL ?? "A";
const speaker2Label =
  process.env.SPEAKER_2_LABEL ?? process.env.STUDENT_SPEAKER_LABEL ?? "B";
const openaiApiKey = process.env.OPENAI_API_KEY;
const hasOpenAIKey =
  typeof openaiApiKey === "string" &&
  openaiApiKey.length > 0 &&
  openaiApiKey !== "your_openai_api_key_here";
const openai = hasOpenAIKey
  ? new OpenAI({
      apiKey: openaiApiKey,
    })
  : null;

type TranscriptRecordStatus = "completed" | "failed";

type TranscriptRecord = {
  id: string;
  status: TranscriptRecordStatus;
  createdAt: string;
  originalFileName?: string;
  savedFileName: string;
  savedPath: string;
  mimeType?: string;
  declaredBytes?: number;
  receivedBytes: number;
  model: string;
  language?: string;
  transcript?: string;
  speakerSegments?: SpeakerSegment[];
  error?: string;
};

type SpeakerSegment = {
  speaker: string;
  role: "話者1" | "話者2" | "話者";
  text: string;
  start?: number;
  end?: number;
  absoluteStart?: string;
  absoluteEnd?: string;
};

type TranscriptionResult = {
  text: string;
  speakerSegments?: SpeakerSegment[];
};

// Parse JSON payloads for future REST endpoints.
app.use(express.json());

// Allow the mobile app to call this backend during development.
app.use(
  cors({
    origin: corsOrigin === "*" ? true : corsOrigin,
  }),
);

// Small logging middleware that records method, path, status, and duration.
app.use((request: Request, response: Response, next: NextFunction) => {
  const startedAt = Date.now();

  response.on("finish", () => {
    const durationMs = Date.now() - startedAt;
    console.log(
      `${request.method} ${request.originalUrl} ${response.statusCode} ${durationMs}ms`,
    );
  });

  next();
});

// Basic health endpoint for local smoke tests and uptime checks.
app.get("/health", (_request: Request, response: Response) => {
  response.status(200).json({
    ok: true,
    service: "music-app-backend",
  });
});

app.get("/status", (_request: Request, response: Response) => {
  response.status(200).json({
    ok: true,
    service: "music-app-backend",
    openaiConfigured: openai !== null,
    transcriptionModel,
    transcriptionLanguage,
    forceJapaneseTranscripts,
    transcriptNormalizerModel,
    uploadDir,
  });
});

app.get("/transcripts", async (request: Request, response: Response) => {
  const limit = Number(request.query.limit ?? 20);

  try {
    const records = await listTranscriptRecords(Number.isFinite(limit) ? limit : 20);

    response.status(200).json({
      items: records,
    });
  } catch (error) {
    response.status(500).json({
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

const server = http.createServer(app);

// WebSocket server mounted manually so only /ws/transcribe is accepted.
const transcribeWss = new WebSocketServer({ noServer: true });

const getRawDataByteLength = (data: RawData): number => {
  // ws can deliver binary messages as Buffer, ArrayBuffer, or Buffer[].
  if (Array.isArray(data)) {
    return data.reduce((total, chunk) => total + chunk.byteLength, 0);
  }

  return data.byteLength;
};

const sanitizeFileName = (fileName: string | undefined): string => {
  const fallbackFileName = `recording-${Date.now()}.mp4`;
  const safeBaseName = path
    .basename(fileName ?? fallbackFileName)
    .replace(/[^a-zA-Z0-9._-]/g, "-");

  return safeBaseName.length > 0 ? safeBaseName : fallbackFileName;
};

const createRecordId = (savedFileName: string): string =>
  savedFileName.replace(/\.[^.]+$/, "");

const sanitizeRecordId = (recordId: string): string =>
  recordId.replace(/[^a-zA-Z0-9._-]/g, "-");

const saveAudioFile = async (
  fileName: string | undefined,
  audioBytes: Buffer,
): Promise<{ savedFileName: string; savedPath: string }> => {
  await mkdir(audioUploadDir, { recursive: true });

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const safeFileName = sanitizeFileName(fileName);
  const savedFileName = `${timestamp}-${safeFileName}`;
  const savedPath = path.join(audioUploadDir, savedFileName);

  await writeFile(savedPath, audioBytes);

  return {
    savedFileName,
    savedPath,
  };
};

const saveTranscriptRecord = async (
  record: TranscriptRecord,
): Promise<{ recordFileName: string; recordPath: string }> => {
  await mkdir(transcriptUploadDir, { recursive: true });

  const recordFileName = `${sanitizeRecordId(record.id)}.json`;
  const recordPath = path.join(transcriptUploadDir, recordFileName);

  await writeFile(recordPath, `${JSON.stringify(record, null, 2)}\n`, "utf8");

  return {
    recordFileName,
    recordPath,
  };
};

const readTranscriptRecord = async (
  recordId: string,
): Promise<TranscriptRecord | null> => {
  const recordPath = path.join(transcriptUploadDir, `${sanitizeRecordId(recordId)}.json`);

  try {
    const fileContents = await readFile(recordPath, "utf8");
    return JSON.parse(fileContents) as TranscriptRecord;
  } catch {
    return null;
  }
};

const mergeTranscriptText = (
  currentTranscript: string | undefined,
  nextTranscript: string | undefined,
): string | undefined => {
  const current = currentTranscript?.trim() ?? "";
  const next = nextTranscript?.trim() ?? "";

  if (current.length === 0) {
    return next.length > 0 ? next : undefined;
  }

  if (next.length === 0) {
    return current;
  }

  return `${current}\n${next}`;
};

const listTranscriptRecords = async (limit: number): Promise<TranscriptRecord[]> => {
  await mkdir(transcriptUploadDir, { recursive: true });

  const entries = await readdir(transcriptUploadDir, { withFileTypes: true });
  const jsonFiles = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
    .map((entry) => entry.name)
    .sort()
    .reverse()
    .slice(0, Math.max(1, Math.min(limit, 100)));

  const records = await Promise.all(
    jsonFiles.map(async (fileName) => {
      const filePath = path.join(transcriptUploadDir, fileName);
      const fileContents = await readFile(filePath, "utf8");

      return JSON.parse(fileContents) as TranscriptRecord;
    }),
  );

  return records;
};

const removeFillers = (text: string): string =>
  text
    .replace(
      /(?:^|[\s、。,.])(?:あー+|えー+|えっと|えと|うーん|うんと|あのー?|そのー?|まぁ|まー)(?=$|[\s、。,.])/g,
      " ",
    )
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\s+([、。,.])/g, "$1")
    .trim();

const shouldNormalizeToJapanese = (text: string): boolean =>
  forceJapaneseTranscripts &&
  /[A-Za-z]{2,}|[가-힣]/.test(text);

const normalizeTranscriptToJapanese = async (text: string): Promise<string> => {
  if (openai === null || !shouldNormalizeToJapanese(text)) {
    return removeFillers(text);
  }

  const cleanedText = removeFillers(text);

  try {
    const response = await openai.chat.completions.create({
      model: transcriptNormalizerModel,
      temperature: 0,
      messages: [
        {
          role: "system",
          content:
            `あなたはサックスレッスンの文字起こしを日本語に整える係です。英語、韓国語、ローマ字、誤変換が混ざっていても、すべて自然な日本語に直してください。[15:34:20] のような時刻、話者1、話者2、話者 の行頭ラベルは必ずそのまま残してください。音楽レッスン文脈を最優先し、次の用語を積極的に採用してください: ${musicTerms}。例えば「scale」は「スケール」、「key」は「キー」、「tone」は「トーン」、「reed」は「リード」、「mouthpiece」は「マウスピース」として扱います。あー、えっと、うーん、あの、その、などのフィラーは削ってください。内容を追加せず、文字起こし本文だけを返してください。`,
        },
        {
          role: "user",
          content: cleanedText,
        },
      ],
    });

    const normalizedText = response.choices[0]?.message.content?.trim();

    return normalizedText !== undefined && normalizedText.length > 0
      ? removeFillers(normalizedText)
      : cleanedText;
  } catch (error) {
    console.error(
      "Failed to normalize transcript to Japanese:",
      error instanceof Error ? error.message : String(error),
    );
    return cleanedText;
  }
};

const getSpeakerRole = (speaker: string): SpeakerSegment["role"] => {
  if (speaker === speaker1Label) {
    return "話者1";
  }

  if (speaker === speaker2Label) {
    return "話者2";
  }

  return "話者";
};

const formatSpeakerTranscript = (segments: SpeakerSegment[]): string => {
  const mergedSegments = segments.reduce<SpeakerSegment[]>((merged, segment) => {
    const previous = merged[merged.length - 1];

    if (previous !== undefined && previous.role === segment.role) {
      previous.text = `${previous.text} ${segment.text}`.trim();
      previous.end = segment.end;
      return merged;
    }

    merged.push({ ...segment });
    return merged;
  }, []);

  return mergedSegments
    .map((segment) => {
      const timestamp =
        typeof segment.absoluteStart === "string"
          ? `[${formatClockTimestamp(segment.absoluteStart)}] `
          : typeof segment.start === "number"
          ? `[${formatTranscriptTimestamp(segment.start)}] `
          : "";

      return `${timestamp}${segment.role}: ${segment.text}`;
    })
    .join("\n");
};

const formatClockTimestamp = (isoTimestamp: string): string => {
  const date = new Date(isoTimestamp);

  if (Number.isNaN(date.getTime())) {
    return isoTimestamp;
  }

  return new Intl.DateTimeFormat("ja-JP", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone: "Asia/Tokyo",
  }).format(date);
};

const formatTranscriptTimestamp = (seconds: number): string => {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const remainingSeconds = safeSeconds % 60;

  if (hours > 0) {
    return [hours, minutes, remainingSeconds]
      .map((value) => value.toString().padStart(2, "0"))
      .join(":");
  }

  return [minutes, remainingSeconds]
    .map((value) => value.toString().padStart(2, "0"))
    .join(":");
};

const offsetSpeakerSegments = (
  segments: SpeakerSegment[] | undefined,
  offsetSeconds: number,
  segmentStartedAt?: string,
): SpeakerSegment[] | undefined => {
  if (segments === undefined || segments.length === 0) {
    return segments;
  }

  const segmentStartedTime =
    segmentStartedAt === undefined ? Number.NaN : new Date(segmentStartedAt).getTime();

  return segments.map((segment) => ({
    ...segment,
    start:
      typeof segment.start === "number" ? segment.start + offsetSeconds : segment.start,
    end: typeof segment.end === "number" ? segment.end + offsetSeconds : segment.end,
    absoluteStart:
      Number.isNaN(segmentStartedTime) || typeof segment.start !== "number"
        ? segment.absoluteStart
        : new Date(segmentStartedTime + segment.start * 1000).toISOString(),
    absoluteEnd:
      Number.isNaN(segmentStartedTime) || typeof segment.end !== "number"
        ? segment.absoluteEnd
        : new Date(segmentStartedTime + segment.end * 1000).toISOString(),
  }));
};

const transcribeAudioFile = async (savedPath: string): Promise<TranscriptionResult> => {
  if (openai === null) {
    throw new Error("OPENAI_API_KEY is not set.");
  }

  const transcriptionRequest: Record<string, unknown> = {
    file: createReadStream(savedPath),
    model: transcriptionModel,
    language: transcriptionLanguage,
  };

  if (transcriptionUsesDiarization) {
    transcriptionRequest.response_format = "diarized_json";
  } else {
    transcriptionRequest.prompt = transcriptionPrompt;
  }

  const transcription = await openai.audio.transcriptions.create(
    transcriptionRequest as never,
  );

  if (transcriptionUsesDiarization) {
    const diarizedTranscription = transcription as unknown as {
      text?: string;
      segments?: Array<{
        speaker?: string;
        text?: string;
        start?: number;
        end?: number;
      }>;
    };
    const speakerSegments =
      diarizedTranscription.segments
        ?.map((segment) => {
          const speaker = segment.speaker ?? "";
          const text = removeFillers(segment.text ?? "");

          return {
            speaker,
            role: getSpeakerRole(speaker),
            text,
            start: segment.start,
            end: segment.end,
          };
        })
        .filter((segment) => segment.text.length > 0) ?? [];

    const rawText =
      speakerSegments.length > 0
        ? formatSpeakerTranscript(speakerSegments)
        : removeFillers(diarizedTranscription.text ?? "");

    return {
      text: await normalizeTranscriptToJapanese(rawText),
      speakerSegments,
    };
  }

  return {
    text: await normalizeTranscriptToJapanese(transcription.text),
  };
};

transcribeWss.on("connection", (socket: WebSocket) => {
  console.log("WebSocket connected: /ws/transcribe");

  socket.send(
    JSON.stringify({
      type: "connection.ready",
      message: "Connected to /ws/transcribe",
    }),
  );

  socket.on("message", async (data) => {
    const messageText = data.toString();

    try {
      const parsedMessage = JSON.parse(messageText) as {
        type?: string;
        fileName?: string;
        mimeType?: string;
        mode?: "replace" | "append";
        sessionId?: string;
        segmentIndex?: number;
        segmentOffsetSeconds?: number;
        segmentStartedAt?: string;
        sizeBytes?: number;
        base64Audio?: string;
      };

      if (parsedMessage.type === "audio.file") {
        const audioBytes = Buffer.from(parsedMessage.base64Audio ?? "", "base64");
        const savedAudio = await saveAudioFile(parsedMessage.fileName, audioBytes);
        const isLiveSession = parsedMessage.mode === "append" && parsedMessage.sessionId;
        const recordId = isLiveSession
          ? sanitizeRecordId(parsedMessage.sessionId ?? "")
          : createRecordId(savedAudio.savedFileName);

        socket.send(
          JSON.stringify({
            type: "audio.file.received",
            fileName: parsedMessage.fileName,
            mimeType: parsedMessage.mimeType,
            declaredBytes: parsedMessage.sizeBytes,
            receivedBytes: audioBytes.byteLength,
            savedFileName: savedAudio.savedFileName,
            savedPath: savedAudio.savedPath,
            message: "Audio file saved. Starting transcription.",
          }),
        );

        try {
          const transcriptionResult = await transcribeAudioFile(savedAudio.savedPath);
          const segmentOffsetSeconds =
            isLiveSession && typeof parsedMessage.segmentOffsetSeconds === "number"
              ? Math.max(0, parsedMessage.segmentOffsetSeconds)
              : isLiveSession && typeof parsedMessage.segmentIndex === "number"
              ? Math.max(0, parsedMessage.segmentIndex - 1) *
                liveTranscriptionChunkSeconds
              : 0;
          const adjustedSpeakerSegments = offsetSpeakerSegments(
            transcriptionResult.speakerSegments,
            segmentOffsetSeconds,
            parsedMessage.segmentStartedAt,
          );
          const transcriptText =
            adjustedSpeakerSegments !== undefined && adjustedSpeakerSegments.length > 0
              ? await normalizeTranscriptToJapanese(
                  formatSpeakerTranscript(adjustedSpeakerSegments),
                )
              : await normalizeTranscriptToJapanese(transcriptionResult.text);
          const existingRecord = isLiveSession
            ? await readTranscriptRecord(recordId)
            : null;
          const existingSegments = existingRecord?.speakerSegments ?? [];
          const nextSegments = adjustedSpeakerSegments ?? [];

          await saveTranscriptRecord({
            id: recordId,
            status: "completed",
            createdAt: existingRecord?.createdAt ?? new Date().toISOString(),
            originalFileName:
              existingRecord?.originalFileName ??
              (isLiveSession ? "リアルタイム文字起こし" : parsedMessage.fileName),
            savedFileName: existingRecord?.savedFileName ?? savedAudio.savedFileName,
            savedPath: savedAudio.savedPath,
            mimeType: parsedMessage.mimeType,
            declaredBytes: parsedMessage.sizeBytes,
            receivedBytes:
              (isLiveSession ? existingRecord?.receivedBytes ?? 0 : 0) +
              audioBytes.byteLength,
            model: transcriptionModel,
            language: transcriptionLanguage,
            transcript: isLiveSession
              ? mergeTranscriptText(existingRecord?.transcript, transcriptText)
              : transcriptText,
            speakerSegments:
              isLiveSession && existingSegments.length + nextSegments.length > 0
                ? [...existingSegments, ...nextSegments]
                : adjustedSpeakerSegments,
          });

          socket.send(
            JSON.stringify({
              type: "audio.transcription.completed",
              id: recordId,
              fileName: parsedMessage.fileName,
              savedFileName: savedAudio.savedFileName,
              model: transcriptionModel,
              language: transcriptionLanguage,
              mode: parsedMessage.mode ?? "replace",
              sessionId: parsedMessage.sessionId,
              segmentIndex: parsedMessage.segmentIndex,
              segmentOffsetSeconds,
              segmentStartedAt: parsedMessage.segmentStartedAt,
              transcript: transcriptText,
              speakerSegments: adjustedSpeakerSegments,
            }),
          );
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          const existingRecord = isLiveSession
            ? await readTranscriptRecord(recordId)
            : null;
          await saveTranscriptRecord({
            id: recordId,
            status: "failed",
            createdAt: existingRecord?.createdAt ?? new Date().toISOString(),
            originalFileName:
              existingRecord?.originalFileName ??
              (isLiveSession ? "リアルタイム文字起こし" : parsedMessage.fileName),
            savedFileName: existingRecord?.savedFileName ?? savedAudio.savedFileName,
            savedPath: savedAudio.savedPath,
            mimeType: parsedMessage.mimeType,
            declaredBytes: parsedMessage.sizeBytes,
            receivedBytes:
              (isLiveSession ? existingRecord?.receivedBytes ?? 0 : 0) +
              audioBytes.byteLength,
            model: transcriptionModel,
            language: transcriptionLanguage,
            transcript: existingRecord?.transcript,
            speakerSegments: existingRecord?.speakerSegments,
            error: errorMessage,
          });

          socket.send(
            JSON.stringify({
              type: "audio.transcription.error",
              id: recordId,
              fileName: parsedMessage.fileName,
              savedFileName: savedAudio.savedFileName,
              mode: parsedMessage.mode ?? "replace",
              sessionId: parsedMessage.sessionId,
              segmentIndex: parsedMessage.segmentIndex,
              message: "Audio file was saved, but transcription failed.",
              error: errorMessage,
            }),
          );
        }

        return;
      }
    } catch (error) {
      if (messageText.trim().startsWith("{")) {
        socket.send(
          JSON.stringify({
            type: "audio.file.error",
            message: "Could not parse or save the audio message.",
            error: error instanceof Error ? error.message : String(error),
          }),
        );
        return;
      }

      // Non-JSON messages continue to the generic placeholder response.
    }

    // This placeholder echoes metadata now; future steps can stream audio here.
    socket.send(
      JSON.stringify({
        type: "transcribe.placeholder",
        receivedBytes: getRawDataByteLength(data),
        message: "Transcription pipeline is not implemented yet.",
      }),
    );
  });

  socket.on("close", () => {
    console.log("WebSocket disconnected: /ws/transcribe");
  });

  socket.on("error", (error: Error) => {
    console.error("WebSocket error on /ws/transcribe:", error.message);
  });
});

server.on("upgrade", (request, socket, head) => {
  const requestUrl = new URL(request.url ?? "/", `http://${request.headers.host}`);

  if (requestUrl.pathname !== "/ws/transcribe") {
    socket.write("HTTP/1.1 404 Not Found\r\n\r\n");
    socket.destroy();
    return;
  }

  transcribeWss.handleUpgrade(request, socket, head, (webSocket) => {
    transcribeWss.emit("connection", webSocket, request);
  });
});

server.listen(port, host, () => {
  console.log(`HTTP server listening at http://${host}:${port}`);
  console.log(`WebSocket endpoint ready at ws://${host}:${port}/ws/transcribe`);
});
