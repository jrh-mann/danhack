import React, { useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { AppState, ChatMessage } from "../types";

type Props = {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  isConnected: boolean;
  isGenerating: boolean;
  state: AppState;
  onUpdateSlider: (id: string, value: number) => void;
};

export function ChatWindow({ messages, onSend, isConnected, isGenerating }: Props) {
  const [input, setInput] = useState("");
  const listRef = useRef<HTMLDivElement | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSend(input);
    setInput("");
  };

  return (
    <div className="flex-1 flex flex-col bg-gradient-to-b from-white to-gray-50 dark:from-neutral-950 dark:to-neutral-900">
      <div ref={listRef} className="flex-1 overflow-auto px-4 lg:px-6 py-6">
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.map((m) => (
            <div key={m.id} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
              <div className={
                "rounded-2xl px-4 py-3 shadow-sm border " +
                (m.role === "user"
                  ? "bg-indigo-600 text-white border-indigo-700"
                  : m.role === "assistant"
                  ? "bg-white dark:bg-neutral-900 border-gray-200 dark:border-neutral-800"
                  : "bg-gray-100 dark:bg-neutral-800 border-gray-200 dark:border-neutral-800")
              }>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                  className={"prose prose-sm max-w-none " + (m.role === "user" ? "prose-invert" : "dark:prose-invert")}
                >
                  {m.content}
                </ReactMarkdown>
              </div>
            </div>
          ))}
          {isGenerating && (
            <div className="text-gray-500 dark:text-gray-400 text-sm">Assistant is typing…</div>
          )}
        </div>
      </div>
      <form onSubmit={handleSubmit} className="sticky bottom-0 border-t border-gray-200 dark:border-neutral-800 bg-white/80 dark:bg-neutral-950/80 backdrop-blur px-4 lg:px-6 py-4">
        <div className="max-w-3xl mx-auto flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Send a message"
            className="flex-1 rounded-xl border border-gray-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <button type="submit" className="px-5 py-3 rounded-xl bg-indigo-600 text-white hover:bg-indigo-700">Send</button>
        </div>
      </form>
    </div>
  );
}


