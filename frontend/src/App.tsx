import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  fetchHealth,
  HealthResponse,
  ResponseMetadata,
  sendChatMessage,
} from "./api";

interface ChatItem {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  metadata?: ResponseMetadata;
}

function generateSessionId(): string {
  const randomPart =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID().slice(0, 12)
      : Math.random().toString(36).slice(2, 14);
  return `s-${randomPart}`;
}

export function App(): React.ReactElement {
  const [sessionId, setSessionId] = useState<string>(() => generateSessionId());
  const [messages, setMessages] = useState<ChatItem[]>([]);
  const [inputText, setInputText] = useState<string>("");
  const [systemPrompt, setSystemPrompt] = useState<string>("");
  const [simulateHeader, setSimulateHeader] = useState<string>("");
  const [isSending, setIsSending] = useState<boolean>(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    fetchHealth()
      .then((h) => setHealth(h))
      .catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    if (
      messagesEndRef.current &&
      typeof messagesEndRef.current.scrollIntoView === "function"
    ) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isSending]);

  const handleNewSession = (): void => {
    setSessionId(generateSessionId());
    setMessages([]);
  };

  const handleSend = async (): Promise<void> => {
    if (isSending) return;
    const rawMessage = inputText;
    setInputText("");
    setIsSending(true);

    const userMsg: ChatItem = {
      id: `user-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      role: "user",
      content: rawMessage,
    };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const res = await sendChatMessage({
        session_id: sessionId,
        message: rawMessage,
        system_prompt: systemPrompt,
        simulate_failure: simulateHeader,
      });

      const assistantMsg: ChatItem = {
        id: `asst-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        role: "assistant",
        content: res.reply,
        metadata: res.metadata,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errText =
        err instanceof Error ? err.message : "An unexpected error occurred.";
      const errMsg: ChatItem = {
        id: `err-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        role: "error",
        content: errText,
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="header-brand">
          <h1>LLM Gateway Chat</h1>
          <span className="subtitle">
            Key Failover &amp; Disk Response Caching
          </span>
        </div>

        <div className="header-controls">
          <div className="session-pill" title="Active in-memory session ID">
            <span className="pill-label">Session:</span>
            <code>{sessionId}</code>
            <button
              type="button"
              className="secondary-btn"
              onClick={handleNewSession}
              disabled={isSending}
            >
              New Session
            </button>
          </div>

          {health && (
            <div className="health-pill">
              <span className="status-dot" />
              <span>{health.keys_configured} Keys</span>
              <span>•</span>
              <span>Cache {health.cache_enabled ? "ON" : "OFF"}</span>
            </div>
          )}
        </div>
      </header>

      <section className="toolbar-bar">
        <div className="toolbar-group">
          <label htmlFor="system-prompt-field">System Prompt</label>
          <input
            id="system-prompt-field"
            data-testid="system-prompt-input"
            type="text"
            placeholder="Optional custom system prompt for this session..."
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            disabled={isSending}
          />
        </div>

        <div className="toolbar-group simulate-group">
          <label htmlFor="simulate-select">Simulate Failure</label>
          <select
            id="simulate-select"
            data-testid="simulate-failure-select"
            value={simulateHeader}
            onChange={(e) => setSimulateHeader(e.target.value)}
            disabled={isSending}
          >
            <option value="">None (Normal operation)</option>
            <option value="key_1=429">key_1=429 (Failover to key_2)</option>
            <option value="key_1=500">key_1=500 (Server error on key_1)</option>
            <option value="key_1=429,key_2=500">
              key_1=429,key_2=500 (Failover to key_3)
            </option>
            <option value="*=429">*=429 (All providers unavailable → 503)</option>
          </select>
        </div>
      </section>

      <main className="chat-container">
        {messages.length === 0 && !isSending ? (
          <div className="empty-state">
            <h2>Start a conversation</h2>
            <p>
              Ask a question to inspect live inference vs. LiteLLM disk caching,
              per-request token usage, and automatic API key failover.
            </p>
          </div>
        ) : (
          <div className="message-list">
            {messages.map((item) => {
              if (item.role === "user") {
                return (
                  <div
                    key={item.id}
                    className="message-row user-row"
                    data-testid="user-message"
                  >
                    <div className="message-bubble user-bubble">
                      <div className="message-text">{item.content}</div>
                    </div>
                  </div>
                );
              }

              if (item.role === "error") {
                return (
                  <div
                    key={item.id}
                    className="message-row error-row"
                    data-testid="error-message"
                    role="alert"
                  >
                    <div className="message-bubble error-bubble">
                      <div className="error-title">Request Error</div>
                      <div className="message-text">{item.content}</div>
                    </div>
                  </div>
                );
              }

              const meta = item.metadata;
              const isCache = meta?.resolution_source === "LLM_CACHE";

              return (
                <div key={item.id} className="message-row assistant-row">
                  <div
                    className="message-bubble assistant-bubble"
                    data-testid="assistant-message"
                  >
                    <div className="message-text markdown-body">
                      <ReactMarkdown>{item.content}</ReactMarkdown>
                    </div>

                    {meta && (
                      <div className="metadata-bar">
                        <span
                          data-testid="meta-source"
                          className={`source-badge ${
                            isCache ? "badge-cache" : "badge-live"
                          }`}
                        >
                          {meta.resolution_source}
                        </span>

                        <span className="meta-chip">
                          <span className="meta-label">Model:</span>{" "}
                          <span data-testid="meta-model">{meta.model}</span>
                        </span>

                        <span className="meta-chip">
                          <span className="meta-label">Latency:</span>{" "}
                          <span data-testid="meta-latency">
                            {meta.latency_ms} ms
                          </span>
                        </span>

                        <span className="meta-chip">
                          <span className="meta-label">In Tokens:</span>{" "}
                          <span data-testid="meta-input-tokens">
                            {meta.input_tokens}
                          </span>
                        </span>

                        <span className="meta-chip">
                          <span className="meta-label">Out Tokens:</span>{" "}
                          <span data-testid="meta-output-tokens">
                            {meta.output_tokens}
                          </span>
                        </span>

                        <span className="meta-chip key-chip">
                          <span className="meta-label">Key:</span>{" "}
                          <span data-testid="meta-key-id">
                            {meta.key_id ?? "none"}
                          </span>
                          {" · "}
                          <span className="meta-label">Fallbacks:</span>{" "}
                          <span data-testid="meta-fallback-attempts">
                            {meta.fallback_attempts}
                          </span>
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {isSending && (
              <div
                className="message-row assistant-row"
                data-testid="thinking-indicator"
              >
                <div className="message-bubble assistant-bubble thinking-bubble">
                  thinking...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      <footer className="composer-bar">
        <div className="composer-inner">
          <textarea
            data-testid="chat-input"
            className="chat-textarea"
            rows={2}
            placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isSending}
          />
          <button
            type="button"
            data-testid="send-button"
            className="primary-btn send-btn"
            onClick={() => void handleSend()}
            disabled={isSending}
          >
            {isSending ? "Sending..." : "Send"}
          </button>
        </div>
      </footer>
    </div>
  );
}

export default App;
