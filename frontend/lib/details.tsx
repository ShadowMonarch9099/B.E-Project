"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";

// "Technical details": off for everyday use. On, the UI shows what the research needs to
// see: file analysis, OCR figures, the router's reasoning, removed sentences, model names.
const KEY = "sakshya.details";

const DetailsContext = createContext<{ on: boolean; toggle: () => void }>({ on: false, toggle: () => {} });

export function DetailsProvider({ children }: { children: React.ReactNode }) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    try {
      setOn(window.localStorage.getItem(KEY) === "1");
    } catch {
      /* storage blocked: stay off */
    }
  }, []);
  const toggle = useCallback(() => {
    setOn((v) => {
      try {
        window.localStorage.setItem(KEY, v ? "0" : "1");
      } catch {
        /* not remembered */
      }
      return !v;
    });
  }, []);
  return <DetailsContext.Provider value={{ on, toggle }}>{children}</DetailsContext.Provider>;
}

export function useDetails() {
  return useContext(DetailsContext);
}

/** Matches a CSS media query; false during server rendering. */
export function useMedia(query: string): boolean {
  const [match, setMatch] = useState(false);
  useEffect(() => {
    const m = window.matchMedia(query);
    const update = () => setMatch(m.matches);
    update();
    m.addEventListener("change", update);
    return () => m.removeEventListener("change", update);
  }, [query]);
  return match;
}
