// Typed client for the RAG API. Shapes mirror app/api/v1/schemas/*.py.

const BASE = "/api/v1";

export type Support = "grounded" | "partial" | "insufficient";

export interface Citation {
  marker: string;
  document_id: string;
  version_id: string;
  chunk_id: string;
  document_title: string | null;
  version_number: number | null;
  page_number: number;
  section_path: string[];
  quote: string | null;
}

export interface EvidencePassage {
  marker: string;
  chunk_id: string;
  document_id: string;
  version_id: string;
  document_title: string | null;
  version_number: number | null;
  page_number: number;
  page_span: number[];
  section_path: string[];
  element_ids: string[];
  text: string;
}

export interface Answer {
  answerId: string | null;
  query: string;
  text: string;
  support: Support;
  abstained: boolean;
  citations: Citation[];
  evidence: EvidencePassage[];
  modelName: string | null;
  latencyMs: number | null;
  degradations: string[];
}

export interface FeedbackInput {
  answer_id: string;
  helpful: boolean;
  answer_correct: boolean | null;
  answer_complete: boolean | null;
  citations_correct: boolean | null;
  source_authoritative: boolean | null;
  comment: string | null;
}

export interface AnswerSummary {
  id: string;
  query: string;
  answer: string;
  support: Support;
  abstained: boolean;
  citations: Citation[];
  evidence: EvidencePassage[];
  model_name: string | null;
  created_at: string;
}

export type FeedbackStatus = "new" | "promoted" | "dismissed";

export interface Feedback extends Omit<FeedbackInput, "answer_id"> {
  id: string;
  answer_id: string;
  status: FeedbackStatus;
  reviewer_note: string | null;
  reviewed_at: string | null;
  candidate_question_id: string | null;
  created_at: string;
  answer: AnswerSummary;
}

export const QUESTION_TYPES = [
  "factual",
  "exact_retrieval",
  "multi_hop",
  "ambiguous",
  "negative_unsupported",
  "temporal",
  "conflicting_versions",
  "calculation",
  "multimodal",
  "adversarial",
] as const;

export interface PromoteInput {
  question_type: (typeof QUESTION_TYPES)[number];
  acceptable_answer: string;
  evidence_markers: string[];
  must_abstain: boolean;
  must_contain: string[];
  split: "dev" | "validation";
  question: string | null;
  reviewer_note: string | null;
}

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail?.[0]?.msg ?? body.detail ?? body.message ?? message;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(typeof message === "string" ? message : JSON.stringify(message));
  }
  return response.json() as Promise<T>;
}

/**
 * Ask a question over SSE. /chat is a POST, so EventSource cannot be used; the
 * stream is read with fetch and split into `event:`/`data:` frames by hand.
 */
export async function askStreaming(
  query: string,
  onUpdate: (partial: Partial<Answer>) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ query, stream: true }),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new ApiError(`The assistant is unavailable (${response.status}).`);
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      handleFrame(frame, onUpdate);
    }
  }
}

function handleFrame(frame: string, onUpdate: (partial: Partial<Answer>) => void): void {
  let event = "message";
  const data: string[] = [];
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return;
  const payload = JSON.parse(data.join("\n"));

  switch (event) {
    case "metadata":
      onUpdate({
        answerId: payload.answer_id ?? null,
        support: payload.support,
        abstained: payload.abstained,
        evidence: payload.evidence ?? [],
        modelName: payload.model_name ?? null,
      });
      break;
    case "token":
      onUpdate({ text: payload.text });
      break;
    case "citations":
      onUpdate({ citations: payload.citations });
      break;
    case "done":
      onUpdate({
        latencyMs: payload.total_latency_ms ?? null,
        degradations: payload.degradations ?? [],
      });
      break;
    case "error":
      throw new ApiError(payload.message ?? "Answer generation failed.");
  }
}

export function submitFeedback(input: FeedbackInput): Promise<Feedback> {
  return request("/feedback", { method: "POST", body: JSON.stringify(input) });
}

export function listFeedback(
  status: FeedbackStatus | "all",
): Promise<{ items: Feedback[]; total: number }> {
  const query = status === "all" ? "" : `?status=${status}`;
  return request(`/feedback${query}`);
}

export function dismissFeedback(id: string, note: string | null): Promise<Feedback> {
  return request(`/feedback/${id}/dismiss`, {
    method: "POST",
    body: JSON.stringify({ reviewer_note: note }),
  });
}

export function promoteFeedback(id: string, input: PromoteInput): Promise<Feedback> {
  return request(`/feedback/${id}/promote`, { method: "POST", body: JSON.stringify(input) });
}

export function reopenFeedback(id: string): Promise<Feedback> {
  return request(`/feedback/${id}/reopen`, { method: "POST" });
}

/** Open the source document at the cited page (Task 13.5). */
export async function openSource(documentId: string, page: number): Promise<void> {
  // Open the tab synchronously so popup blockers treat it as user-initiated.
  const tab = window.open("about:blank", "_blank");
  try {
    const { presigned_url } = await request<{ presigned_url: string }>(
      `/documents/${documentId}/presigned-url`,
    );
    const url = `${presigned_url}#page=${page}`;
    if (tab) tab.location.href = url;
    else window.location.href = url;
  } catch (error) {
    tab?.close();
    throw error;
  }
}

export function citationLabel(c: {
  document_title: string | null;
  version_number: number | null;
  page_number: number;
}): string {
  const title = (c.document_title ?? "Unknown document").replace(/\.(pdf|docx?|pptx?|xlsx?)$/i, "");
  const version = c.version_number ? ` · v${c.version_number}` : "";
  return `${title}${version} · p.${c.page_number}`;
}
