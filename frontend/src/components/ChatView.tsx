import { useEffect, useRef, useState, type FormEvent } from "react";
import { askStreaming, citationLabel, openSource, type Answer } from "../api";
import { AnswerText } from "./AnswerText";
import { EvidencePanel } from "./EvidencePanel";
import { FeedbackForm } from "./FeedbackForm";

interface Turn {
  id: number;
  query: string;
  answer: Answer;
  status: "streaming" | "done" | "error";
  error?: string;
}

const EXAMPLES = [
  "How many days of annual leave do new employees get?",
  "What is the notice period during probation?",
  "Can I carry unused leave into next year?",
];

function emptyAnswer(query: string): Answer {
  return {
    answerId: null,
    query,
    text: "",
    support: "grounded",
    abstained: false,
    citations: [],
    evidence: [],
    modelName: null,
    latencyMs: null,
    degradations: [],
  };
}

const SUPPORT_LABEL: Record<Answer["support"], string> = {
  grounded: "Grounded in policy",
  partial: "Partially supported",
  insufficient: "Not in the policy corpus",
};

export function ChatView() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [selected, setSelected] = useState<number | null>(null);
  const [activeMarker, setActiveMarker] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const busy = turns.some((t) => t.status === "streaming");

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns.length]);

  async function ask(query: string) {
    const id = Date.now();
    setTurns((ts) => [...ts, { id, query, answer: emptyAnswer(query), status: "streaming" }]);
    setSelected(id);
    setActiveMarker(null);
    setInput("");

    const update = (patch: Partial<Turn>) =>
      setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...patch } : t)));
    try {
      await askStreaming(query, (partial) =>
        setTurns((ts) =>
          ts.map((t) => (t.id === id ? { ...t, answer: { ...t.answer, ...partial } } : t)),
        ),
      );
      update({ status: "done" });
    } catch (error) {
      update({
        status: "error",
        error: error instanceof Error ? error.message : "Something went wrong.",
      });
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    const query = input.trim();
    if (query && !busy) void ask(query);
  }

  function selectMarker(turnId: number, marker: string) {
    setSelected(turnId);
    setActiveMarker(marker);
  }

  const selectedTurn = turns.find((t) => t.id === selected) ?? null;

  return (
    <div className="chat-layout">
      <section className="conversation">
        <div className="turns">
          {turns.length === 0 && (
            <div className="welcome">
              <h1>Ask about HR policy</h1>
              <p className="muted">
                Answers come only from the company's policy documents, with a citation for every
                claim. When the documents don't cover a question, the assistant says so.
              </p>
              <div className="examples">
                {EXAMPLES.map((q) => (
                  <button key={q} type="button" onClick={() => void ask(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {turns.map((turn) => (
            <article
              key={turn.id}
              className={`turn${turn.id === selected ? " is-selected" : ""}`}
              onClick={() => setSelected(turn.id)}
            >
              <div className="question">{turn.query}</div>
              <TurnBody
                turn={turn}
                activeMarker={turn.id === selected ? activeMarker : null}
                onMarker={(m) => selectMarker(turn.id, m)}
              />
            </article>
          ))}
          <div ref={bottom} />
        </div>

        <form className="composer" onSubmit={submit}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) submit(e);
            }}
            placeholder="Ask a question about leave, benefits, conduct…"
            rows={2}
            maxLength={4000}
            aria-label="Question"
          />
          <button type="submit" className="primary" disabled={busy || !input.trim()}>
            {busy ? "Answering…" : "Ask"}
          </button>
        </form>
      </section>

      <EvidencePanel
        answer={selectedTurn?.status === "error" ? null : (selectedTurn?.answer ?? null)}
        activeMarker={activeMarker}
        onMarker={(m) => selected !== null && selectMarker(selected, m)}
      />
    </div>
  );
}

function TurnBody({
  turn,
  activeMarker,
  onMarker,
}: {
  turn: Turn;
  activeMarker: string | null;
  onMarker: (marker: string) => void;
}) {
  const { answer } = turn;
  const [openError, setOpenError] = useState(false);

  if (turn.status === "error") {
    return (
      <div className="answer is-error" role="alert">
        <strong>The assistant could not answer.</strong>
        <p>{turn.error}</p>
      </div>
    );
  }
  if (turn.status === "streaming" && !answer.text) {
    return (
      <div className="answer is-pending">
        <span className="dots" aria-hidden>
          <i />
          <i />
          <i />
        </span>
        Searching the policy documents…
      </div>
    );
  }

  const markers = new Set(answer.citations.map((c) => c.marker));

  return (
    <div className={`answer${answer.abstained ? " is-abstained" : ""}`}>
      <div className={`support support-${answer.support}`}>
        {answer.abstained && <span aria-hidden>ⓘ </span>}
        {SUPPORT_LABEL[answer.support]}
      </div>

      <AnswerText
        text={answer.text}
        activeMarker={activeMarker}
        knownMarkers={markers}
        onMarker={onMarker}
      />

      {answer.abstained && (
        <p className="abstain-hint">
          This is a deliberate refusal, not an error: nothing in the policy corpus supports an
          answer. If you believe it should, tell us below.
        </p>
      )}

      {answer.citations.length > 0 && (
        <div className="citations">
          {answer.citations.map((c) => (
            <button
              key={c.marker}
              type="button"
              className={`chip${activeMarker === c.marker ? " is-active" : ""}`}
              onClick={(e) => {
                e.stopPropagation();
                onMarker(c.marker);
              }}
              onDoubleClick={() =>
                void openSource(c.document_id, c.page_number).catch(() => setOpenError(true))
              }
              title={`${c.section_path.join(" › ")}\nDouble-click to open the source at this page`}
            >
              <span className="marker static">{c.marker}</span>
              {citationLabel(c)}
            </button>
          ))}
        </div>
      )}
      {openError && <p className="error small">The source document could not be opened.</p>}

      {turn.status === "done" && (
        <div className="answer-foot">
          <span className="muted small">
            {answer.modelName ?? "model unknown"}
            {answer.latencyMs !== null && ` · ${(answer.latencyMs / 1000).toFixed(1)} s`}
          </span>
          {answer.answerId ? (
            <FeedbackForm answerId={answer.answerId} />
          ) : (
            <span className="muted small">This answer was not recorded, so it cannot be rated.</span>
          )}
        </div>
      )}
    </div>
  );
}
