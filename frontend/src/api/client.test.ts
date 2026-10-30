import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, setAdminToken } from "./client";

function respond(status: number, body: unknown) {
  return vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
}

afterEach(() => {
  vi.unstubAllGlobals();
  setAdminToken("");
});

describe("api client", () => {
  it("sends the admin token only on admin calls", async () => {
    const fetchMock = respond(200, []);
    vi.stubGlobal("fetch", fetchMock);
    setAdminToken("s3cret");
    await api.conversations(5);
    await api.voiceOffer({ sdp: "v=0", type: "offer" });
    const [adminCall, voiceCall] = fetchMock.mock.calls as unknown as [[string, RequestInit], [string, RequestInit]];
    expect(adminCall[0]).toBe("/admin/conversations?limit=5");
    expect(new Headers(adminCall[1].headers).get("Authorization")).toBe("Bearer s3cret");
    expect(new Headers(voiceCall[1].headers).get("Authorization")).toBeNull();
    expect(voiceCall[1].method).toBe("POST");
  });

  it("turns errors into ApiError with the server's detail", async () => {
    vi.stubGlobal("fetch", respond(503, { detail: "Voice isn't available" }));
    await expect(api.voiceOffer({ sdp: "", type: "offer" })).rejects.toEqual(new ApiError(503, "Voice isn't available"));
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("network"); }));
    await expect(api.latestEval()).rejects.toMatchObject({ status: 0 });
  });

  it("escapes path parameters", async () => {
    const fetchMock = respond(200, {});
    vi.stubGlobal("fetch", fetchMock);
    await api.evalRun("../secrets");
    expect((fetchMock.mock.calls[0] as unknown as [string])[0]).toBe("/admin/evals/..%2Fsecrets");
  });
});
