import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { Aperture, BookOpen, Camera, CarFront, Check, ChevronDown, CircleAlert, FileUp, LoaderCircle, MessageSquare, Plus, Send, ShieldCheck, X } from 'lucide-react'
import { api } from './api'

const initialVehicle = { manufacturer: 'Hyundai', model: 'i20', year: 2024 }
const readable = value => (value || '').replaceAll('_', ' ')

function AssistantMessage({ message }) {
  return <article className="assistant-message">
    <div className="message-author"><span className="avatar"><Aperture size={17} /></span> AutoLens <span className="muted">Manual-grounded guidance</span></div>
    <div className="markdown"><ReactMarkdown>{message.content}</ReactMarkdown></div>
    {message.severity && <div className={`severity severity-${message.severity.level.toLowerCase()}`}>
      <div className="flex items-center gap-2 font-semibold"><ShieldCheck size={18} />{message.severity.level} · Safety guidance</div>
      <p>{message.severity.recommended_action}</p>
      <small>Troubleshooting guidance only — not a replacement for a qualified mechanic.</small>
    </div>}
    {!!message.citations?.length && <div className="citations">{message.citations.map((citation, index) =>
      <details key={`${citation.source}-${citation.page_number}-${index}`}>
        <summary><BookOpen size={14} /> View source · Page {citation.page_number}<ChevronDown size={14} /></summary>
        <div><strong>{citation.source}</strong><p>{citation.text}</p></div>
      </details>)}</div>}
  </article>
}

export default function App() {
  const [vehicles, setVehicles] = useState([])
  const [vehicleId, setVehicleId] = useState('')
  const [vehicle, setVehicle] = useState(initialVehicle)
  const [sessions, setSessions] = useState([])
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [manualStatus, setManualStatus] = useState('')
  const [vision, setVision] = useState(null)
  const bottom = useRef(null)
  const historyGeneration = useRef(0)

  useEffect(() => {
    let active = true
    api('/vehicles').then(data => { if (active) setVehicles(data.vehicles) }).catch(err => { if (active) setError(err.message) })
    return () => { active = false }
  }, [])

  useEffect(() => {
    const generation = ++historyGeneration.current
    api(`/history/sessions${vehicleId ? `?vehicle_id=${encodeURIComponent(vehicleId)}` : ''}`)
      .then(data => { if (generation === historyGeneration.current) setSessions(data.sessions) })
      .catch(err => { if (generation === historyGeneration.current) setError(err.message) })
  }, [vehicleId])

  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, busy])

  function newChat() {
    setSessionId(null); setMessages([]); setDraft(''); setVision(null); setError('')
  }

  function selectVehicle(id) {
    const selected = vehicles.find(item => item.id === id)
    setVehicleId(id)
    setVehicle(selected || initialVehicle)
    setManualStatus(selected?.manual_filename ? `Ready · ${selected.manual_filename}` : '')
    newChat()
  }

  async function refreshHistory(id) {
    const data = await api(`/history/sessions${id ? `?vehicle_id=${encodeURIComponent(id)}` : ''}`)
    setSessions(data.sessions)
  }

  async function uploadManual(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    if (!vehicle.manufacturer.trim() || !vehicle.model.trim() || !vehicle.year) {
      setError('Enter the manufacturer, model and year before uploading a manual.'); return
    }
    setBusy('Indexing your manual'); setError(''); setManualStatus('Reading and indexing your PDF…')
    const form = new FormData()
    form.append('file', file)
    for (const key of ['manufacturer', 'model', 'year']) form.append(key, vehicle[key])
    try {
      const result = await api('/rag/ingest', { method: 'POST', body: form })
      setVehicleId(result.vehicle_id)
      setManualStatus(`Ready · ${result.source} · ${result.pages_extracted} pages`)
      newChat()
      const data = await api('/vehicles')
      setVehicles(data.vehicles)
    } catch (err) { setError(err.message); setManualStatus('Upload failed. Please try again.') }
    finally { setBusy('') }
  }

  async function loadSession(id) {
    setBusy('Loading conversation'); setError('')
    try {
      const session = await api(`/history/sessions/${encodeURIComponent(id)}`)
      const selected = vehicles.find(item => item.id === session.vehicle_id)
      setSessionId(id); setMessages(session.messages); setDraft(''); setVision(null)
      setVehicleId(session.vehicle_id || '')
      setVehicle(selected || initialVehicle)
      setManualStatus(selected?.manual_filename ? `Ready · ${selected.manual_filename}` : '')
    } catch (err) { setError(err.message) } finally { setBusy('') }
  }

  async function sendMessage(event) {
    event.preventDefault()
    const text = draft.trim()
    if (!text || busy) return
    setBusy('Reading your manual'); setError('')
    try {
      const result = await api('/chat', { json: {
        message: text, session_id: sessionId, ...(vehicleId ? { vehicle_id: vehicleId } : sessionId ? {} : {
          manufacturer: vehicle.manufacturer, model: vehicle.model, year: Number(vehicle.year),
        }), warning_type: vision?.status === 'detected' ? vision.warning : null,
      } })
      setSessionId(result.session_id)
      setMessages(previous => [...previous, { role: 'user', content: text },
        { role: 'assistant', content: result.reply, severity: result.severity, citations: result.citations }])
      setDraft(''); setVision(null)
      // Refresh auxiliary lists independently: a history failure must not resend a completed turn.
      try {
        const data = await api('/vehicles')
        setVehicles(data.vehicles)
        const selected = data.vehicles.find(item => item.manufacturer === vehicle.manufacturer && item.model === vehicle.model && item.year === Number(vehicle.year))
        const id = vehicleId || selected?.id || ''
        if (id) setVehicleId(id)
        await refreshHistory(id)
      } catch (err) { setError(`Message saved. Could not refresh history: ${err.message}`) }
    } catch (err) { setError(err.message) } finally { setBusy('') }
  }

  async function uploadImage(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    if (file.size > 10 * 1024 * 1024) { setError('Please choose an image smaller than 10 MB.'); return }
    setBusy('Examining dashboard photo'); setError(''); setVision(null)
    const form = new FormData(); form.append('file', file)
    try {
      const result = await api('/vision/analyze', { method: 'POST', body: form })
      setVision(result)
      if (result.status === 'detected') setDraft(`My dashboard shows the ${readable(result.warning)}. What does my manual say?`)
    } catch (err) { setError(err.message) } finally { setBusy('') }
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="./"><Aperture size={31} strokeWidth={1.7} />AutoLens<span className="brand-dot">.</span></a>
      <p className="sidebar-caption">A little clarity for the road ahead.</p>
      <h2>My vehicles & history</h2>
      <fieldset disabled={!!busy} className="vehicle-fields">
        <label>Your garage<select value={vehicleId} onChange={e => selectVehicle(e.target.value)}>
          <option value="">Add a vehicle</option>{vehicles.map(item => <option key={item.id} value={item.id}>{item.manufacturer} {item.model} · {item.year}</option>)}
        </select></label>
        <div className="vehicle-card"><CarFront size={27} /><div><strong>{vehicle.manufacturer || 'Your'} {vehicle.model || 'vehicle'}</strong><span>{vehicle.year || 'Select a model year'}</span></div></div>
        {['manufacturer', 'model', 'year'].map(key => <label key={key} className="capitalize">{key}
          <input type={key === 'year' ? 'number' : 'text'} min={key === 'year' ? 1980 : undefined} max={key === 'year' ? 2027 : undefined}
            maxLength={50} value={vehicle[key]} readOnly={!!vehicleId} onChange={e => { setVehicle({ ...vehicle, [key]: e.target.value }); setManualStatus(''); newChat() }} />
        </label>)}
        <label className="upload-manual"><FileUp size={18} />Upload owner's manual<input className="sr-only" type="file" accept=".pdf,application/pdf" onChange={uploadManual} /></label>
      </fieldset>
      <p className="manual-status" role="status">{manualStatus ? <><Check size={14} />{manualStatus}</> : <><BookOpen size={14} />PDF manual · Answers specific to your car</>}</p>
      <div className="history-heading"><h2>Conversations</h2><button className="icon-button" title="New conversation" aria-label="New conversation" disabled={!!busy} onClick={newChat}><Plus size={19} /></button></div>
      <nav aria-label="Past conversations" className="history-list">{sessions.length ? sessions.map(session =>
        <button key={session.id} disabled={!!busy} className={sessionId === session.id ? 'selected' : ''} onClick={() => loadSession(session.id)}>
          <MessageSquare size={16} /><span>{session.title || 'Diagnostic conversation'}<small>{new Date(session.updated_at).toLocaleDateString()} · {session.severity || 'New'}</small></span>
        </button>) : <p className="empty-history">Your conversations will appear here.</p>}</nav>
      <div className="sidebar-footer"><ShieldCheck size={18} /><span>Your manual. Clearer answers.</span></div>
    </aside>

    <main className="workspace">
      <header className="workspace-header"><div><span className="eyebrow">YOUR MANUAL, MADE HELPFUL</span><h1>Vehicle assistant</h1></div><span className="grounded-tag"><BookOpen size={15} />Manual-grounded</span></header>
      <div className="chat-scroll">
        {!messages.length && <section className="welcome">
          <div className="welcome-symbol"><Aperture size={42} strokeWidth={1.3} /></div>
          <span className="eyebrow">LET'S MAKE SENSE OF IT</span>
          <h2>See the warning.<br /><span>Understand the problem.</span></h2>
          <p>Upload your owner's manual, then tell me what you're seeing.<br className="desktop-break" /> We'll find the relevant guidance, one page at a time.</p>
          <div className="starter-grid">
            <button disabled={!!busy} onClick={() => setDraft('My engine warning light is on. What should I check?')}><CircleAlert size={22} /><strong>A warning light is on</strong><span>Understand what your manual says</span></button>
            <button disabled={!!busy} onClick={() => document.getElementById('dashboard-image').click()}><Camera size={22} /><strong>Identify a dashboard symbol</strong><span>Start with a clear photo</span></button>
          </div>
          <div className="welcome-note"><BookOpen size={16} />Every answer starts with your owner's manual.</div>
        </section>}
        <div className="message-stream" aria-live="polite">{messages.map((message, index) => message.role === 'user'
          ? <article key={message.id || index} className="user-message"><span>You</span><p>{message.content}</p></article>
          : <AssistantMessage key={message.id || index} message={message} />)}
          {busy && <div className="loading" role="status"><LoaderCircle className="animate-spin" size={18} />{busy}…</div>}
          <div ref={bottom} />
        </div>
      </div>
      <div className="composer-area">
        {error && <div className="error-banner" role="alert"><CircleAlert size={18} /><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error"><X size={16} /></button></div>}
        {vision && <div className={`vision-result ${vision.status}`} role="status">{vision.status === 'detected'
          ? <><Camera size={17} /><strong className="capitalize">{readable(vision.warning)}</strong><span>{Math.round(vision.confidence * 100)}% confidence</span></>
          : <><CircleAlert size={17} /><span>{vision.prompt}</span></>}<button disabled={!!busy} aria-label="Clear image result" onClick={() => setVision(null)}><X size={15} /></button></div>}
        <form className="composer" onSubmit={sendMessage}>
          <label className={`camera-button ${busy ? 'disabled' : ''}`} title="Upload dashboard photo"><Camera size={22} /><span className="sr-only">Upload dashboard photo</span><input id="dashboard-image" className="sr-only" disabled={!!busy} type="file" accept="image/jpeg,image/png,image/webp" onChange={uploadImage} /></label>
          <textarea aria-label="Message AutoLens" placeholder="Describe a warning light or ask about your vehicle…" rows={2} value={draft} disabled={!!busy} onChange={e => setDraft(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); e.currentTarget.form.requestSubmit() } }} />
          <button className="send-button" aria-label="Send message" disabled={!draft.trim() || !!busy}><Send size={19} /></button>
        </form>
        <p className="composer-note">Guidance from your manual. Always follow professional advice and drive safely.</p>
      </div>
    </main>
  </div>
}
