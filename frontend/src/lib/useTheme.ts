import { useEffect, useState } from "react";

export function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    try {
      const ls = localStorage.getItem("theme");
      if (ls === "light" || ls === "dark") return ls;
    } catch {}
    return "dark";
  });

  useEffect(() => {
    const root = document.documentElement.classList;
    if (theme === "dark") root.add("dark"); else root.remove("dark");
    try { localStorage.setItem("theme", theme); } catch {}
  }, [theme]);

  const toggle = () => setTheme((t) => (t === "dark" ? "light" : "dark"));
  return { theme, toggle };
}


