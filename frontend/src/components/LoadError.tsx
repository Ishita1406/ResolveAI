import { useState } from "react";
import { type ApiError, getAdminToken, setAdminToken } from "../api/client";

/** An API error. A 401/403 asks for the admin token (needed beyond localhost when the
 * backend sets ADMIN_TOKEN) and retries. */
export function LoadError({ error, retry }: { error: ApiError; retry: () => void }) {
  const [token, setToken] = useState(getAdminToken());
  const needsToken = error.status === 401 || error.status === 403;
  return (
    <div className="notice error" role="alert">
      <p>{error.message}</p>
      {needsToken ? (
        <form
          className="token-form"
          style={{ marginTop: 10 }}
          onSubmit={(e) => {
            e.preventDefault();
            setAdminToken(token.trim());
            retry();
          }}
        >
          <input
            type="password"
            placeholder="Admin token"
            aria-label="Admin token"
            autoComplete="off"
            value={token}
            onChange={(e) => setToken(e.target.value)}
          />
          <button className="btn small" type="submit">Use token</button>
        </form>
      ) : (
        <button className="btn ghost small" style={{ marginTop: 10 }} onClick={retry}>Try again</button>
      )}
    </div>
  );
}
