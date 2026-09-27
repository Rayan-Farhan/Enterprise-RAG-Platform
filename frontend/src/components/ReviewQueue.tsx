import { useCallback, useEffect, useRef, useState } from "react";
import {
  QUESTION_TYPES,
  citationLabel,
  dismissFeedback,
  listFeedback,
  promoteFeedback,
  reopenFeedback,
  type Feedback,
  type FeedbackStatus,
  type PromoteInput,
} from "../api";

const FILTERS: (FeedbackStatus | "all")[] = ["new", "promoted", "dismissed", "all"];

const JUDGEMENTS: { key: keyof Feedback; label: string }[] = [
  { key: "answer_correct", label: "Correct" },
  { key: "answer_complete", label: "Complete" },
  { key: "citations_correct", label: "Citations" },
  { key: "source_authoritative", label: "Right source" },
];

/**
 * The review queue (Task 13.4): triage feedback, then either dismiss it or turn
 * it into a golden-dataset candidate. Candidates enter a split only after a
 * maintainer runs `eval accept-candidates` — the second human gate.
 */
export function ReviewQueue() {
  const [filter, setFilter] = useState<FeedbackStatus | "all">("new");
  const [items, setItems] = useState<Feedback[]>([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  // A review action finishing after the user switched filters must reload the
  // *current* filter, so reloads bump a counter instead of calling a captured
  // loader. And only the newest request may update the list.
  const [reloads, setReloads] = useState(0);
  const latest = useRef(0);

  const load = useCallback(async () => {
    const request = ++latest.current;
    setLoading(true);
    setError(null);
    try {
      const page = await listFeedback(filter);
      if (request !== latest.current) return;
      setItems(page.items);
      setTotal(page.total);
    } catch (e) {
      if (request !== latest.current) return;
      setError(e instanceof Error ? e.message : "Could not load the queue.");
    } finally {
      if (request === latest.current) setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    void load();
  }, [load, reloads]);

  const reload = useCallback(() => setReloads((n) => n + 1), []);

  return (
    <div className="review">
      <header className="review-head">
        <div>
          <h1>Review queue</h1>
          <p className="muted small">
            Promoted feedback becomes a golden-dataset candidate. Accept candidates into a split
            with <code>python -m app.evaluation.cli accept-candidates</code>.
          </p>
        </div>
        <div className="segmented" role="tablist">
          {FILTERS.map((f) => (
            <button
              key={f}
              type="button"
              role="tab"
              aria-selected={filter === f}
              className={filter === f ? "is-on" : ""}
              onClick={() => setFilter(f)}
            >
              {f}
            </button>
          ))}
        </div>
      </header>

      {error && <p className="error">{error}</p>}
      {!loading && !error && items.length === 0 && (
        <p className="muted empty">Nothing here. Feedback from the chat lands in “new”.</p>
      )}
      <p className="muted small">{loading ? "Loading…" : `${total} item${total === 1 ? "" : "s"}`}</p>

      {items.map((item) => (
        <ReviewItem key={item.id} item={item} onChanged={reload} />
      ))}
    </div>
  );
}

function ReviewItem({ item, onChanged }: { item: Feedback; onChanged: () => void }) {
  const [mode, setMode] = useState<"view" | "promote" | "dismiss">("view");
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const { answer } = item;

  async function act(fn: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Action failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <article className={`review-item status-${item.status}`}>
      <div className="review-meta">
        <span className={`rating ${item.helpful ? "up" : "down"}`}>
          {item.helpful ? "👍" : "👎"}
        </span>
        <span className={`status-pill ${item.status}`}>{item.status}</span>
        <span className="muted small">{new Date(item.created_at).toLocaleString()}</span>
        {item.candidate_question_id && (
          <code className="small">{item.candidate_question_id}</code>
        )}
      </div>

      <div className="question">{answer.query}</div>
      <div className={`answer compact${answer.abstained ? " is-abstained" : ""}`}>
        <div className={`support support-${answer.support}`}>{answer.support}</div>
        <p className="prewrap">{answer.answer}</p>
        {answer.citations.length > 0 && (
          <div className="citations">
            {answer.citations.map((c) => (
              <span key={c.marker} className="chip static">
                <span className="marker static">{c.marker}</span>
                {citationLabel(c)}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="judgements">
        {JUDGEMENTS.map(({ key, label }) => {
          const value = item[key] as boolean | null;
          return (
            <span key={key} className={`judged ${value === null ? "na" : value ? "yes" : "no"}`}>
              {label}: {value === null ? "—" : value ? "yes" : "no"}
            </span>
          );
        })}
      </div>
      {item.comment && <blockquote>{item.comment}</blockquote>}
      {item.reviewer_note && <p className="small">Reviewer: {item.reviewer_note}</p>}

      {error && <p className="error small">{error}</p>}

      {item.status === "new" && mode === "view" && (
        <div className="review-actions">
          <button type="button" className="primary" onClick={() => setMode("promote")}>
            Promote to evaluation case
          </button>
          <button type="button" onClick={() => setMode("dismiss")}>
            Dismiss
          </button>
        </div>
      )}
      {item.status !== "new" && (
        <div className="review-actions">
          <button type="button" disabled={busy} onClick={() => void act(() => reopenFeedback(item.id))}>
            Reopen
          </button>
        </div>
      )}

      {mode === "dismiss" && (
        <div className="review-form">
          <textarea
            placeholder="Why no evaluation case is needed (optional)"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            rows={2}
          />
          <div className="review-actions">
            <button
              type="button"
              className="primary"
              disabled={busy}
              onClick={() => void act(() => dismissFeedback(item.id, note.trim() || null))}
            >
              Confirm dismiss
            </button>
            <button type="button" onClick={() => setMode("view")}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {mode === "promote" && (
        <PromoteForm
          item={item}
          busy={busy}
          onCancel={() => setMode("view")}
          onSubmit={(input) => void act(() => promoteFeedback(item.id, input))}
        />
      )}
    </article>
  );
}

function PromoteForm({
  item,
  busy,
  onCancel,
  onSubmit,
}: {
  item: Feedback;
  busy: boolean;
  onCancel: () => void;
  onSubmit: (input: PromoteInput) => void;
}) {
  const { answer } = item;
  const [questionType, setQuestionType] = useState<PromoteInput["question_type"]>(
    answer.abstained ? "negative_unsupported" : "factual",
  );
  const [reference, setReference] = useState("");
  const [markers, setMarkers] = useState<string[]>(answer.citations.map((c) => c.marker));
  const [mustAbstain, setMustAbstain] = useState(false);
  const [mustContain, setMustContain] = useState("");
  const [split, setSplit] = useState<PromoteInput["split"]>("dev");
  const [note, setNote] = useState("");

  const abstentionType = questionType === "negative_unsupported" || questionType === "adversarial";
  const noEvidence = abstentionType || mustAbstain;

  function toggle(marker: string) {
    setMarkers((ms) => (ms.includes(marker) ? ms.filter((m) => m !== marker) : [...ms, marker]));
  }

  return (
    <form
      className="review-form"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit({
          question_type: questionType,
          acceptable_answer: reference.trim(),
          evidence_markers: noEvidence ? [] : markers,
          must_abstain: mustAbstain,
          must_contain: mustContain
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
          split,
          question: null,
          reviewer_note: note.trim() || null,
        });
      }}
    >
      <div className="form-grid">
        <label>
          Question type
          <select
            value={questionType}
            onChange={(e) => setQuestionType(e.target.value as PromoteInput["question_type"])}
          >
            {QUESTION_TYPES.map((t) => (
              <option key={t} value={t}>
                {t.replace(/_/g, " ")}
              </option>
            ))}
          </select>
        </label>
        <label>
          Target split
          <select value={split} onChange={(e) => setSplit(e.target.value as PromoteInput["split"])}>
            <option value="dev">dev</option>
            <option value="validation">validation</option>
          </select>
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={mustAbstain}
            onChange={(e) => setMustAbstain(e.target.checked)}
          />
          The correct behaviour is to refuse
        </label>
      </div>

      <label>
        Reference answer
        <textarea
          required
          value={reference}
          onChange={(e) => setReference(e.target.value)}
          rows={2}
          placeholder="What a correct answer says"
        />
      </label>

      <label>
        Must contain (comma-separated, optional)
        <input value={mustContain} onChange={(e) => setMustContain(e.target.value)} />
      </label>

      {!noEvidence && (
        <fieldset>
          <legend>Which passages answer the question?</legend>
          {answer.evidence.length === 0 && (
            <p className="muted small">
              The model was shown no passages. An answerable case needs evidence, so the right
              passage has to be found by re-running the question first.
            </p>
          )}
          {answer.evidence.map((e) => (
            <label key={e.marker} className="evidence-option">
              <input
                type="checkbox"
                checked={markers.includes(e.marker)}
                onChange={() => toggle(e.marker)}
              />
              <span>
                <strong>
                  [{e.marker}] {citationLabel(e)}
                </strong>
                <span className="passage-text clamp">{e.text}</span>
              </span>
            </label>
          ))}
        </fieldset>
      )}

      <label>
        Reviewer note (optional)
        <input value={note} onChange={(e) => setNote(e.target.value)} />
      </label>

      <div className="review-actions">
        <button type="submit" className="primary" disabled={busy || !reference.trim()}>
          Create candidate
        </button>
        <button type="button" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}
