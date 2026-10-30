import { useEffect, useState } from "react";
import { AdminPage } from "./admin/AdminPage";
import { CallPage } from "./call/CallPage";
import { EvalsPage } from "./evals/EvalsPage";

// Hash routes, so any static server (Vite, nginx) serves every page from index.html:
//   #/                 call          #/conversations[/42]      admin replay
//   #/evals[/<run>]    eval results
export type Route =
  | { page: "call" }
  | { page: "conversations"; id: number | null }
  | { page: "evals"; run: string | null };

export function parseRoute(hash: string): Route {
  const [page, arg] = hash.replace(/^#\/?/, "").split("/");
  if (page === "conversations") {
    const id = Number(arg);
    return { page, id: arg && Number.isInteger(id) && id > 0 ? id : null };
  }
  if (page === "evals") return { page, run: arg ? decodeURIComponent(arg) : null };
  return { page: "call" };
}

function href(route: Route): string {
  switch (route.page) {
    case "call":
      return "#/";
    case "conversations":
      return route.id ? `#/conversations/${route.id}` : "#/conversations";
    case "evals":
      return route.run ? `#/evals/${encodeURIComponent(route.run)}` : "#/evals";
  }
}

function useRoute(): Route {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

const go = (route: Route) => {
  window.location.hash = href(route);
};

export function App() {
  const route = useRoute();
  const nav = [
    { page: "call", label: "Call" },
    { page: "conversations", label: "Conversations" },
    { page: "evals", label: "Evals" },
  ] as const;

  return (
    <div className="shell">
      <header className="topbar">
        <a className="brand" href="#/">
          <span className="brand-dot" aria-hidden="true" />
          ResolveAI
        </a>
        <nav className="nav" aria-label="Main">
          {nav.map((item) => (
            <a
              key={item.page}
              href={`#/${item.page === "call" ? "" : item.page}`}
              aria-current={route.page === item.page ? "page" : undefined}
            >
              {item.label}
            </a>
          ))}
        </nav>
      </header>
      <main>
        {route.page === "call" && <CallPage />}
        {route.page === "conversations" && (
          <AdminPage selected={route.id} onSelect={(id) => go({ page: "conversations", id })} />
        )}
        {route.page === "evals" && <EvalsPage run={route.run} onRun={(run) => go({ page: "evals", run })} />}
      </main>
    </div>
  );
}
