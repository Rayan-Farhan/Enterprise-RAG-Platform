import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { ChatView } from "./components/ChatView";
import { ReviewQueue } from "./components/ReviewQueue";
import "./styles.css";

type View = "chat" | "review";

// Hash routing: two views do not justify a router dependency.
function currentView(): View {
  return window.location.hash === "#/review" ? "review" : "chat";
}

function App() {
  const [view, setView] = useState<View>(currentView);

  useEffect(() => {
    const onHash = () => setView(currentView());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>
            ◆
          </span>
          HR Policy Assistant
        </div>
        <nav>
          <a href="#/" className={view === "chat" ? "is-on" : ""}>
            Ask
          </a>
          <a href="#/review" className={view === "review" ? "is-on" : ""}>
            Review queue
          </a>
        </nav>
      </header>
      {/* Both stay mounted so the conversation survives a trip to the queue. */}
      <main hidden={view !== "chat"}>
        <ChatView />
      </main>
      {view === "review" && (
        <main>
          <ReviewQueue />
        </main>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
