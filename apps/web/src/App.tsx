import { useState } from "react";

import { AIEvidence } from "./components/AIEvidence";
import { Workbench } from "./components/Workbench";

type View = "workbench" | "ai";

export default function App() {
  const [view, setView] = useState<View>("workbench");

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => setView("workbench")}><span className="brand__mark">AQ</span><span><strong>AgentQAKit</strong><small>Agent quality workspace</small></span></button>
        <nav aria-label="Primary"><button className={view === "workbench" ? "active" : ""} onClick={() => setView("workbench")}>Workspace</button><button className={view === "ai" ? "active" : ""} onClick={() => setView("ai")}>AI activity</button></nav>
        <div className="live-indicator"><span /> Local</div>
      </header>

      <main>
        {view === "workbench" && <Workbench />}
        {view === "ai" && <AIEvidence />}
      </main>
      <footer><span>AgentQAKit v0.1 alpha</span><span>Local evidence stays under your control</span></footer>
    </div>
  );
}
