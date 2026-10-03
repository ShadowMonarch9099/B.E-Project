import type { Conversation, DocDetail, DocSummary, Health, PageText, UploadResult } from "./types";

async function readError(res: Response, fallback: string): Promise<Error> {
  const body = await res.json().catch(() => ({}));
  const detail = typeof body.detail === "string" ? body.detail : null;
  return new Error(detail ?? `${fallback} (${res.status}).`);
}

async function getJson<T>(url: string, what: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, { cache: "no-store" });
  } catch {
    throw new Error("Could not reach the backend. Is it running on port 8000?");
  }
  if (!res.ok) throw await readError(res, `Could not load ${what}`);
  return res.json();
}

export function getHealth(): Promise<Health> {
  return getJson("/api/health", "the status");
}

export async function listDocuments(): Promise<DocSummary[]> {
  return (await getJson<{ documents: DocSummary[] }>("/api/documents", "your documents")).documents;
}

export function getDocument(id: string): Promise<DocDetail> {
  return getJson(`/api/documents/${id}`, "the document");
}

export async function getPages(id: string): Promise<PageText[]> {
  return (await getJson<{ pages: PageText[] }>(`/api/documents/${id}/pages`, "the pages")).pages;
}

export function fileUrl(id: string): string {
  return `/api/documents/${id}/file`;
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await fetch(`/api/documents/${id}`, { method: "DELETE" });
  if (!res.ok && res.status !== 404) throw await readError(res, "Could not delete the document");
}

export async function clearAllDocuments(): Promise<number> {
  const res = await fetch("/api/documents", { method: "DELETE" });
  if (!res.ok) throw await readError(res, "Could not clear the documents");
  return (await res.json()).deleted;
}

export async function reindexDocument(id: string): Promise<DocSummary> {
  const res = await fetch(`/api/documents/${id}/reindex`, { method: "POST" });
  if (!res.ok) throw await readError(res, "Could not re-index the document");
  return res.json();
}

// XMLHttpRequest rather than fetch, because fetch cannot report upload progress.
export function uploadDocument(file: File, onProgress: (fraction: number) => void): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file, file.name);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onerror = () => reject(new Error("Could not reach the backend. Is it running on port 8000?"));
    xhr.onload = () => {
      let body: Record<string, unknown> = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        return reject(new Error(`The server returned an unreadable response (${xhr.status}).`));
      }
      if (xhr.status === 422 && body.detection) {
        return resolve({ kind: "rejected", detail: String(body.detail), detection: body.detection as never });
      }
      if (xhr.status >= 400) {
        return reject(new Error(typeof body.detail === "string" ? body.detail : `Upload failed (${xhr.status}).`));
      }
      resolve({ kind: "stored", ...(body as Omit<Extract<UploadResult, { kind: "stored" }>, "kind">) });
    };
    xhr.open("POST", "/api/documents");
    xhr.send(form);
  });
}

// ---------- conversations ----------

export async function latestConversation(docId: string): Promise<Conversation | null> {
  const list = await getJson<{ conversations: Conversation[] }>(
    `/api/conversations?doc_id=${encodeURIComponent(docId)}`,
    "conversations",
  );
  if (list.conversations.length === 0) return null;
  return getJson<Conversation>(`/api/conversations/${list.conversations[0].id}`, "the conversation");
}

export async function createConversation(docId: string): Promise<Conversation> {
  let res: Response;
  try {
    res = await fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: docId }),
    });
  } catch {
    throw new Error("Could not reach the backend. Is it running on port 8000?");
  }
  if (!res.ok) throw await readError(res, "Could not start a conversation");
  return res.json();
}

export interface AskHandlers {
  onEvent: (event: string, data: Record<string, unknown>) => void;
  signal?: AbortSignal;
}

// POST a question and read the Server-Sent Events it returns. EventSource cannot POST,
// so the stream is read with fetch and parsed here.
export async function askQuestion(conversationId: string, question: string, { onEvent, signal }: AskHandlers) {
  let res: Response;
  try {
    res = await fetch(`/api/conversations/${conversationId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ question }),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    throw new Error("Could not reach the backend. Is it running on port 8000?");
  }
  if (!res.ok || !res.body) throw await readError(res, "Could not ask the question");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data.push(line.slice(6));
      }
      if (data.length) onEvent(event, JSON.parse(data.join("\n")));
    }
  }
}
