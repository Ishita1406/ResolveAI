import { useCallback, useEffect, useState } from "react";
import { ApiError } from "../api/client";

export interface Loaded<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  reload: () => void;
}

/** Run `load` when `key` changes; keeps the last error for display. */
export function useLoad<T>(load: (() => Promise<T>) | null, key: unknown): Loaded<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!load) return;
    let current = true;
    setLoading(true);
    setError(null);
    load()
      .then((value) => current && setData(value))
      .catch((e: unknown) => {
        if (!current) return;
        setData(null);
        setError(e instanceof ApiError ? e : new ApiError(0, String(e)));
      })
      .finally(() => current && setLoading(false));
    return () => {
      current = false;
    };
    // `load` is a new closure each render; `key` is what names the request.
  }, [key, tick]);

  return { data, error, loading, reload: useCallback(() => setTick((t) => t + 1), []) };
}
