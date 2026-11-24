const BASE_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

export async function api(path, { json, ...options } = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    ...(json !== undefined ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(json) } : {}),
  })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map(item => item.msg).join('; ') : data.detail
    throw new Error(detail || `Request failed (${response.status}). Please try again.`)
  }
  return data
}
