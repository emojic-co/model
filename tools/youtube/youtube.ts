// Manage the @emojify-ing YouTube channel via the Data API v3.
// Credentials: youtube.json (OAuth desktop client, gitignored); token cache: youtube.token.json (gitignored).
// Usage: bun tools/youtube/youtube.ts <auth|status|banner <png>|describe <text|@file>|keywords <a,b,c>|upload <video> <title> [desc]|publish <videoId>|edit <videoId> <title> <desc|@file>|list>
// Note: the API cannot change the profile picture; upload that in YouTube Studio.
import { existsSync } from "node:fs"

const CLIENT = "youtube.json"
const TOKEN = "youtube.token.json"
const SCOPES = ["https://www.googleapis.com/auth/youtube", "https://www.googleapis.com/auth/youtube.upload"]
const API = "https://www.googleapis.com"

type Client = { client_id: string; client_secret: string; token_uri: string; auth_uri: string }
type Token = { access_token: string; refresh_token: string; expires_at: number }

const client = async (): Promise<Client> => (await Bun.file(CLIENT).json()).installed

async function tokenRequest(c: Client, body: Record<string, string>): Promise<any> {
  const res = await fetch(c.token_uri, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ client_id: c.client_id, client_secret: c.client_secret, ...body }),
  })
  const json = await res.json()
  if (!res.ok) throw new Error(`token endpoint: ${JSON.stringify(json)}`)
  return json
}

async function auth(): Promise<void> {
  const c = await client()
  const server = Bun.serve({ port: 0, fetch: (req) => { onCode(new URL(req.url).searchParams.get("code")); return new Response("Done, you can close this tab.") } })
  let onCode!: (code: string | null) => void
  const code = new Promise<string | null>((r) => (onCode = r))
  const redirect = `http://127.0.0.1:${server.port}`
  const url = `${c.auth_uri}?${new URLSearchParams({
    client_id: c.client_id, redirect_uri: redirect, response_type: "code",
    scope: SCOPES.join(" "), access_type: "offline", prompt: "consent",
  })}`
  console.log(`Open this URL and approve access:\n\n${url}\n`)
  const got = await code
  server.stop()
  if (!got) throw new Error("no auth code received")
  const t = await tokenRequest(c, { code: got, redirect_uri: redirect, grant_type: "authorization_code" })
  await Bun.write(TOKEN, JSON.stringify({ access_token: t.access_token, refresh_token: t.refresh_token, expires_at: Date.now() + t.expires_in * 1000 } satisfies Token, null, 2))
  console.log(`saved ${TOKEN}`)
}

async function accessToken(): Promise<string> {
  if (!existsSync(TOKEN)) throw new Error("not authorized: run `bun tools/youtube/youtube.ts auth`")
  const t: Token = await Bun.file(TOKEN).json()
  if (t.expires_at > Date.now() + 60_000) return t.access_token
  const r = await tokenRequest(await client(), { refresh_token: t.refresh_token, grant_type: "refresh_token" })
  await Bun.write(TOKEN, JSON.stringify({ ...t, access_token: r.access_token, expires_at: Date.now() + r.expires_in * 1000 }, null, 2))
  return r.access_token
}

async function api(path: string, init: RequestInit & { json?: unknown } = {}): Promise<any> {
  const headers = new Headers(init.headers)
  headers.set("authorization", `Bearer ${await accessToken()}`)
  if (init.json !== undefined) { headers.set("content-type", "application/json"); init.body = JSON.stringify(init.json) }
  const res = await fetch(`${API}${path}`, { ...init, headers })
  const text = await res.text()
  if (!res.ok) throw new Error(`${res.status} ${path}: ${text}`)
  return text ? JSON.parse(text) : {}
}

async function channel(): Promise<any> {
  const r = await api("/youtube/v3/channels?part=snippet,brandingSettings,statistics&mine=true")
  if (!r.items?.length) throw new Error("no channel for this account")
  return r.items[0]
}

async function updateBranding(mutate: (ch: any) => void): Promise<void> {
  const ch = await channel()
  const body = { id: ch.id, brandingSettings: ch.brandingSettings }
  mutate(body)
  await api("/youtube/v3/channels?part=brandingSettings", { method: "PUT", json: body })
}

const [cmd, ...args] = process.argv.slice(2)

switch (cmd) {
  case "auth": await auth(); break
  case "status": {
    const ch = await channel()
    console.log(JSON.stringify({ id: ch.id, title: ch.snippet.title, customUrl: ch.snippet.customUrl, stats: ch.statistics,
      description: ch.brandingSettings.channel?.description, keywords: ch.brandingSettings.channel?.keywords,
      banner: ch.brandingSettings.image?.bannerExternalUrl }, null, 2))
    break
  }
  case "banner": {
    const [path] = args
    if (!path) throw new Error("usage: banner <png>")
    const file = Bun.file(path)
    const up = await api("/upload/youtube/v3/channelBanners/insert", { method: "POST", headers: { "content-type": file.type || "image/png" }, body: file })
    await updateBranding((b) => { b.brandingSettings.image = { ...b.brandingSettings.image, bannerExternalUrl: up.url } })
    console.log("banner updated")
    break
  }
  case "describe": {
    const [text] = args
    if (!text) throw new Error("usage: describe <text|@file>")
    const description = text.startsWith("@") ? await Bun.file(text.slice(1)).text() : text
    await updateBranding((b) => { b.brandingSettings.channel = { ...b.brandingSettings.channel, description } })
    console.log("description updated")
    break
  }
  case "keywords": {
    const [list] = args
    if (!list) throw new Error("usage: keywords a,b,c")
    const keywords = list.split(",").map((k) => `"${k.trim()}"`).join(" ")
    await updateBranding((b) => { b.brandingSettings.channel = { ...b.brandingSettings.channel, keywords } })
    console.log("keywords updated")
    break
  }
  case "list": {
    const ch = await channel()
    const r = await api(`/youtube/v3/search?part=snippet&forMine=true&type=video&maxResults=50&order=date`)
    for (const v of r.items) console.log(v.id.videoId, v.snippet.title)
    void ch
    break
  }
  case "edit": {
    const [id, title, text] = args
    if (!id || !title || !text) throw new Error("usage: edit <videoId> <title> <desc|@file>")
    const description = text.startsWith("@") ? await Bun.file(text.slice(1)).text() : text
    const cur = await api(`/youtube/v3/videos?part=snippet&id=${id}`)
    if (!cur.items?.length) throw new Error(`no such video: ${id}`)
    await api("/youtube/v3/videos?part=snippet", { method: "PUT", json: { id, snippet: { ...cur.items[0].snippet, title, description } } })
    console.log(`updated: https://youtu.be/${id}`)
    break
  }
  case "publish": {
    const [id] = args
    if (!id) throw new Error("usage: publish <videoId>")
    const cur = await api(`/youtube/v3/videos?part=status&id=${id}`)
    if (!cur.items?.length) throw new Error(`no such video: ${id}`)
    await api("/youtube/v3/videos?part=status", { method: "PUT", json: { id, status: { ...cur.items[0].status, privacyStatus: "public" } } })
    console.log(`public: https://youtu.be/${id}`)
    break
  }
  case "upload": {
    const [path, title, description = ""] = args
    if (!path || !title) throw new Error("usage: upload <video> <title> [desc]  (uploads as private)")
    const file = Bun.file(path)
    const init = await fetch(`${API}/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status`, {
      method: "POST",
      headers: { authorization: `Bearer ${await accessToken()}`, "content-type": "application/json", "x-upload-content-type": file.type || "video/mp4" },
      body: JSON.stringify({ snippet: { title, description }, status: { privacyStatus: "private" } }),
    })
    if (!init.ok) throw new Error(`${init.status}: ${await init.text()}`)
    const res = await fetch(init.headers.get("location")!, { method: "PUT", headers: { "content-type": file.type || "video/mp4" }, body: file })
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)
    console.log(`uploaded (private): https://youtu.be/${(await res.json()).id}`)
    break
  }
  default:
    console.error("commands: auth | status | banner <png> | describe <text|@file> | keywords <a,b,c> | list | publish <videoId> | edit <videoId> <title> <desc|@file> | upload <video> <title> [desc]")
    process.exit(1)
}
