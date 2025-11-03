import React from "react";
import { useTheme } from "../lib/useTheme";

export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle dark mode"
      className="rounded-md border border-gray-300 dark:border-neutral-700 px-3 py-1.5 text-sm bg-white dark:bg-neutral-900 hover:bg-gray-50 dark:hover:bg-neutral-800"
    >
      {theme === "dark" ? "Light" : "Dark"}
    </button>
  );
}


