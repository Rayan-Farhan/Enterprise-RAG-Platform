import { useEffect, useRef, useState } from "react";
import { citationLabel, openSource, type Answer } from "../api";

interface Props {
  answer: Answer | null;
  activeMarker: string | null;
  onMarker: (marker: string) => void;
}

/** Every passage the model was given, with the cited ones marked. */
export function EvidencePanel({ answer, activeMarker, onMarker }: Props) {
  const refs = useRef(new Map<string, HTMLElement>());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (activeMarker) {
      refs.current.get(activeMarker)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [activeMarker, answer]);

  if (!answer) {
    return (
      <aside className="evidence">
        <h2>Evidence</h2>
        <p className="muted">The passages behind an answer appear here.</p>
      </aside>
    );
  }

  const cited = new Set(answer.citations.map((c) => c.marker));

  async function open(documentId: string, page: number) {
    setError(null);
    try {
      await openSource(documentId, page);
    } catch {
      setError("The source document could not be opened. Is object storage running?");
    }
  }

  return (
    <aside className="evidence">
      <h2>
        Evidence <span className="count">{answer.evidence.length}</span>
      </h2>
      {answer.evidence.length === 0 ? (
        <p className="muted">
          {answer.abstained
            ? "No passage in the policy corpus was relevant enough to answer from."
            : "No evidence was recorded for this answer."}
        </p>
      ) : (
        <p className="muted small">
          {cited.size} of {answer.evidence.length} passages cited. The model saw all of them.
        </p>
      )}
      {error && <p className="error small">{error}</p>}
      <ol className="passages">
        {answer.evidence.map((e) => (
          <li
            key={e.marker}
            ref={(el) => {
              if (el) refs.current.set(e.marker, el);
            }}
            className={[
              "passage",
              cited.has(e.marker) ? "is-cited" : "",
              activeMarker === e.marker ? "is-active" : "",
            ].join(" ")}
            onClick={() => onMarker(e.marker)}
          >
            <div className="passage-head">
              <span className="marker static">{e.marker}</span>
              <button
                type="button"
                className="link"
                onClick={(ev) => {
                  ev.stopPropagation();
                  void open(e.document_id, e.page_number);
                }}
                title="Open the source document at this page"
              >
                {citationLabel(e)} ↗
              </button>
              {!cited.has(e.marker) && <span className="tag">not cited</span>}
            </div>
            {e.section_path.length > 0 && (
              <div className="section-path">{e.section_path.join(" › ")}</div>
            )}
            <p className="passage-text">{e.text}</p>
          </li>
        ))}
      </ol>
    </aside>
  );
}
