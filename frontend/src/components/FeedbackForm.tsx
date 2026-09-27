import { useState } from "react";
import { submitFeedback, type FeedbackInput } from "../api";

type Judgement = boolean | null;

const QUESTIONS: { key: keyof FeedbackInput; label: string }[] = [
  { key: "answer_correct", label: "Was the answer correct?" },
  { key: "answer_complete", label: "Was it complete?" },
  { key: "citations_correct", label: "Did the citations support it?" },
  { key: "source_authoritative", label: "Was the source the right policy?" },
];

/**
 * Structured feedback (Task 13.3). A thumb alone says something went wrong; the
 * judgements say what, which is what a reviewer needs to turn it into a test case.
 * Unanswered judgements are sent as null — "not assessed", not "no".
 */
export function FeedbackForm({ answerId }: { answerId: string }) {
  const [helpful, setHelpful] = useState<boolean | null>(null);
  const [judgements, setJudgements] = useState<Record<string, Judgement>>({});
  const [comment, setComment] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "sent" | "error">("idle");

  if (state === "sent") {
    return <div className="feedback sent">Thanks — your feedback is in the review queue.</div>;
  }

  async function send(overrideHelpful?: boolean) {
    const rating = overrideHelpful ?? helpful;
    if (rating === null) return;
    setState("sending");
    try {
      await submitFeedback({
        answer_id: answerId,
        helpful: rating,
        answer_correct: judgements.answer_correct ?? null,
        answer_complete: judgements.answer_complete ?? null,
        citations_correct: judgements.citations_correct ?? null,
        source_authoritative: judgements.source_authoritative ?? null,
        comment: comment.trim() || null,
      });
      setState("sent");
    } catch {
      setState("error");
    }
  }

  return (
    <div className="feedback">
      <div className="feedback-row">
        <span className="muted small">Was this helpful?</span>
        <button
          type="button"
          className={`thumb${helpful === true ? " is-on" : ""}`}
          onClick={() => setHelpful(true)}
          aria-pressed={helpful === true}
        >
          👍 Yes
        </button>
        <button
          type="button"
          className={`thumb${helpful === false ? " is-on" : ""}`}
          onClick={() => setHelpful(false)}
          aria-pressed={helpful === false}
        >
          👎 No
        </button>
      </div>

      {helpful !== null && (
        <div className="feedback-detail">
          {QUESTIONS.map((q) => (
            <div className="judgement" key={q.key}>
              <span>{q.label}</span>
              <div className="segmented" role="radiogroup" aria-label={q.label}>
                {(
                  [
                    [true, "Yes"],
                    [false, "No"],
                    [null, "Not sure"],
                  ] as const
                ).map(([value, text]) => (
                  <button
                    key={text}
                    type="button"
                    role="radio"
                    aria-checked={(judgements[q.key] ?? null) === value}
                    className={(judgements[q.key] ?? null) === value ? "is-on" : ""}
                    onClick={() => setJudgements({ ...judgements, [q.key]: value })}
                  >
                    {text}
                  </button>
                ))}
              </div>
            </div>
          ))}
          <textarea
            placeholder={helpful ? "Anything to add? (optional)" : "What went wrong? (optional)"}
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            rows={2}
            maxLength={4000}
          />
          <div className="feedback-actions">
            {state === "error" && <span className="error small">Could not send. Try again.</span>}
            <button
              type="button"
              className="primary"
              disabled={state === "sending"}
              onClick={() => void send()}
            >
              {state === "sending" ? "Sending…" : "Send feedback"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
