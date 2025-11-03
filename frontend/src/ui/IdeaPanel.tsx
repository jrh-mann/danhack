import React, { useState } from "react";
import { AppState } from "../types";

type Props = {
  state: AppState;
  onSubmitIdea: (idea: string) => void;
  onUpdateSlider: (id: string, value: number) => void;
};

export function IdeaPanel({ state, onSubmitIdea, onUpdateSlider }: Props) {
  const [idea, setIdea] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmitIdea(idea);
    setIdea("");
  };

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-neutral-900 border border-gray-200 dark:border-neutral-800 rounded-xl p-4">
        <h2 className="font-semibold mb-3">Steering Idea</h2>
        <form onSubmit={handleSubmit} className="space-y-3">
          <textarea
            value={idea}
            onChange={(e) => setIdea(e.target.value)}
            placeholder="Describe your steering idea"
            rows={4}
            className="w-full rounded-lg border border-gray-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <button type="submit" className="w-full rounded-lg bg-indigo-600 text-white py-2.5 hover:bg-indigo-700">Submit Idea</button>
        </form>
      </div>

      <div className="bg-white dark:bg-neutral-900 border border-gray-200 dark:border-neutral-800 rounded-xl p-4">
        <h3 className="font-medium mb-3">Sliders</h3>
        <div className="space-y-4">
          {state.sliders.length === 0 && (
            <div className="text-sm text-gray-500 dark:text-gray-400">No sliders yet. Submit an idea to get started.</div>
          )}
          {state.sliders.map((s) => (
            <div key={s.id} className="rounded-lg border border-gray-200 dark:border-neutral-800 p-3">
              <div className="flex justify-between mb-1 text-sm">
                <div className="font-medium">{s.label}</div>
                <div className="text-gray-500 dark:text-gray-400">{s.value.toFixed(2)}</div>
              </div>
              <input
                type="range"
                min={s.min}
                max={s.max}
                step={s.step}
                value={s.value}
                onChange={(e) => onUpdateSlider(s.id, parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="relative h-1 mt-1">
                <div className="absolute left-1/2 -translate-x-1/2 top-0 bottom-0 w-px bg-gray-300 dark:bg-neutral-700"></div>
              </div>
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                <span>{s.min}</span>
                <span>0</span>
                <span>{s.max}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


