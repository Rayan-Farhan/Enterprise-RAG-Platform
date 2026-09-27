import type { ReactNode } from "react";

// A deliberately small renderer: paragraphs, bullet lists, **bold**, and [n]
// citation markers as buttons. The model is prompted for plain prose; pulling in
// a full Markdown library for that would be the heaviest thing in the bundle.

interface Props {
  text: string;
  activeMarker: string | null;
  knownMarkers: Set<string>;
  onMarker: (marker: string) => void;
}

export function AnswerText({ text, activeMarker, knownMarkers, onMarker }: Props) {
  const blocks = text.trim().split(/\n{2,}/);
  return (
    <div className="answer-text">
      {blocks.map((block, i) => {
        const lines = block.split("\n");
        const isList = lines.every((l) => /^\s*([-*]|\d+\.)\s+/.test(l));
        if (isList) {
          return (
            <ul key={i}>
              {lines.map((l, j) => (
                <li key={j}>{inline(l.replace(/^\s*([-*]|\d+\.)\s+/, ""))}</li>
              ))}
            </ul>
          );
        }
        return (
          <p key={i}>
            {lines.map((l, j) => (
              <span key={j}>
                {j > 0 && <br />}
                {inline(l)}
              </span>
            ))}
          </p>
        );
      })}
    </div>
  );

  function inline(line: string): ReactNode[] {
    return line.split(/(\*\*[^*]+\*\*|\[\d+\])/g).map((part, k) => {
      const marker = part.match(/^\[(\d+)\]$/)?.[1];
      if (marker && knownMarkers.has(marker)) {
        return (
          <button
            key={k}
            type="button"
            className={`marker${activeMarker === marker ? " is-active" : ""}`}
            onClick={() => onMarker(marker)}
            title={`Show evidence [${marker}]`}
          >
            {marker}
          </button>
        );
      }
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={k}>{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  }
}
