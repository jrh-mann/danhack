import React, { useEffect, useMemo, useRef, useState } from "react";
import { ChatMessage, AppState, ServerEvent, SendChatPayload, SendIdeaPayload, SliderSpec } from "../types";
import { useWebSocketClient } from "../lib/useWebSocketClient";
import { ChatWindow } from "./ChatWindow";
import { IdeaPanel } from "./IdeaPanel";
import { ThemeToggle } from "./ThemeToggle";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws";

export function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [state, setState] = useState<AppState>({ sliders: [] });
  const [isGenerating, setIsGenerating] = useState(false);

  const { isOpen, sendJson, subscribe } = useWebSocketClient(WS_URL);

  useEffect(() => {
    const unsubscribe = subscribe((evt) => {
      try {
        const data = JSON.parse(evt.data) as ServerEvent;
        if (data.type === "assistant_token") {
          setIsGenerating(true);
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (!last || last.role !== "assistant") {
              return [...prev, { id: crypto.randomUUID(), role: "assistant", content: data.token }];
            }
            const cloned = [...prev];
            cloned[cloned.length - 1] = { ...last, content: last.content + data.token };
            return cloned;
          });
        } else if (data.type === "assistant_done") {
          setIsGenerating(false);
        } else if (data.type === "system") {
          setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "system", content: data.content }]);
        } else if (data.type === "add_slider") {
          setState((s) => ({ ...s, sliders: [...s.sliders, data.slider] }));
        }
      } catch {
        // ignore malformed
      }
    });
    return unsubscribe;
  }, [subscribe]);

  const handleSendChat = (text: string) => {
    if (!text.trim()) return;
    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    const payload: SendChatPayload = { type: "chat", message: text, state };
    sendJson(payload);
  };

  const handleSubmitIdea = (idea: string) => {
    if (!idea.trim()) return;
    const payload: SendIdeaPayload = { type: "idea", idea, state };
    sendJson(payload);
  };

  const updateSlider = (id: string, value: number) => {
    setState((s) => ({
      ...s,
      sliders: s.sliders.map((sl) => (sl.id === id ? { ...sl, value } : sl))
    }));
  };

  return (
    <div className="h-full bg-white dark:bg-neutral-950">
      <header className="sticky top-0 z-10 border-b border-gray-200 dark:border-neutral-800 bg-white/80 dark:bg-neutral-950/80 backdrop-blur">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="font-semibold">Chatbot</div>
          <div className="flex items-center gap-3">
            <div className="text-xs text-gray-600 dark:text-gray-400">
              <span className={"inline-block w-2 h-2 rounded-full mr-2 " + (isOpen ? "bg-indigo-500" : "bg-gray-400")}></span>
              {isOpen ? "Connected" : "Disconnected"}
            </div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="h-[calc(100%-57px)] grid grid-cols-1 lg:grid-cols-[1fr_380px]">
        <div className="border-r border-gray-200 dark:border-neutral-800 flex flex-col">
          <ChatWindow
            messages={messages}
            onSend={handleSendChat}
            isConnected={isOpen}
            isGenerating={isGenerating}
            state={state}
            onUpdateSlider={updateSlider}
          />
        </div>
        <div className="p-4 lg:p-6 overflow-auto">
          <IdeaPanel state={state} onSubmitIdea={handleSubmitIdea} onUpdateSlider={updateSlider} />
        </div>
      </main>
    </div>
  );
}


