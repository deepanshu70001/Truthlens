import { useState, useCallback, useEffect, useRef } from 'react';
import {
  Shield, Upload, AlertTriangle, CheckCircle, XCircle, Clock, Loader2,
  Eye, FileText, Globe, Search, TrendingUp, MapPin, Users, Info,
  Zap, AlertOctagon, HelpCircle
} from 'lucide-react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { MapContainer, TileLayer, Tooltip as MapTooltip, CircleMarker } from 'react-leaflet';
import { uploadImage, getAnalysisStatus, getAnalysisReport, getHeatmapUrl, getAnalysisImageUrl } from './services/api';
import './index.css';

/* ────── helpers ────── */
const pct = (v?: number) => v != null ? `${(v * 100).toFixed(1)}%` : '—';
const clsx = (...cls: (string | false | undefined)[]) => cls.filter(Boolean).join(' ');

const STEPS = [
  'Preprocessing', 'Metadata', 'Quality', 'Authenticity', 'Forensics', 'OCR',
  'Vision', 'XAI', 'Text Normalization', 'Claim Extraction', 'Fact Check',
  'Provenance', 'Hashing', 'Evidence Collection', 'Fusion', 'Explanation',
  'Virality', 'Regional Spread', 'Social Impact', 'Report Assembly'
];

const STEP_MAP: Record<string, number> = {};
STEPS.forEach((s, i) => STEP_MAP[s.toLowerCase().replace(/\s+/g, '_')] = i);

/* ────── CARD ────── */
function Card({ title, icon: Icon, children, className, badge }: {
  title: string; icon?: any; children: React.ReactNode; className?: string; badge?: React.ReactNode;
}) {
  return (
    <div className={clsx('bg-surface-2/80 backdrop-blur-sm border border-border rounded-2xl p-6', className)}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          {Icon && <Icon className="w-5 h-5 text-primary-light" />}
          <h3 className="text-lg font-semibold text-text">{title}</h3>
        </div>
        {badge}
      </div>
      {children}
    </div>
  );
}

/* ────── STATUS BADGE ────── */
function StatusBadge({ status }: { status: string }) {
  const cfg: Record<string, { color: string; label: string }> = {
    DATA_UNAVAILABLE: { color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30', label: 'Data Unavailable' },
    DEMO: { color: 'bg-purple-500/20 text-purple-400 border-purple-500/30', label: 'Demo Data' },
    unavailable: { color: 'bg-gray-500/20 text-gray-400 border-gray-500/30', label: 'Unavailable' },
    failed: { color: 'bg-red-500/20 text-red-400 border-red-500/30', label: 'Failed' },
    skipped: { color: 'bg-gray-500/20 text-gray-400 border-gray-500/30', label: 'Skipped' },
    partial: { color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30', label: 'Partial' },
    completed: { color: 'bg-green-500/20 text-green-400 border-green-500/30', label: 'Completed' },
    running: { color: 'bg-blue-500/20 text-blue-400 border-blue-500/30', label: 'Running' },
    success: { color: 'bg-green-500/20 text-green-400 border-green-500/30', label: 'Available' },
    no_results: { color: 'bg-gray-500/20 text-gray-400 border-gray-500/30', label: 'No Results' },
  };
  const c = cfg[status] || cfg['unavailable'];
  return <span className={`text-xs px-2.5 py-1 rounded-full border ${c.color}`}>{c.label}</span>;
}

/* ────── PROGRESS TRACKER ────── */
function ProgressTracker({ completed, current, failed }: {
  completed: string[]; current?: string; failed?: string[];
}) {
  const norm = (s: string) => s.toLowerCase().replace(/\s+/g, '_');
  const cSet = new Set((completed || []).map(norm));
  const fSet = new Set((failed || []).map(norm));
  const cur = current ? norm(current) : '';

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
      {STEPS.map(step => {
        const key = norm(step);
        let icon = <Clock className="w-3.5 h-3.5 text-text-muted" />;
        let color = 'text-text-muted';
        if (cSet.has(key)) { icon = <CheckCircle className="w-3.5 h-3.5 text-success" />; color = 'text-success'; }
        else if (fSet.has(key)) { icon = <XCircle className="w-3.5 h-3.5 text-danger" />; color = 'text-danger'; }
        else if (key === cur) { icon = <Loader2 className="w-3.5 h-3.5 text-primary-light animate-spin" />; color = 'text-primary-light'; }
        return (
          <div key={step} className={`flex items-center gap-2 px-3 py-2 rounded-lg bg-surface/50 ${color}`}>
            {icon} <span className="text-xs font-medium truncate">{step}</span>
          </div>
        );
      })}
    </div>
  );
}

/* ────── AUTHENTICITY SECTION ────── */
function AuthenticitySection({ data }: { data: any }) {
  const prediction = data?.model_prediction;
  const label = prediction === 'likely_ai_generated' || prediction === 'likely_authentic'
    ? prediction
    : typeof data?.ai_generated_score === 'number' && typeof data?.authentic_score === 'number'
      ? data.ai_generated_score > data.authentic_score ? 'likely_ai_generated' : 'likely_authentic'
      : data?.label;
  const cfg = label === 'likely_ai_generated'
    ? { color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/30', text: 'AI-Generated' }
    : label === 'likely_authentic'
      ? { color: 'text-green-400', bg: 'bg-green-500/10 border-green-500/30', text: 'Real / Authentic' }
      : null;
  const scores = [
    { name: 'AI-generated raw score', value: data?.ai_generated_score },
    { name: 'Real / authentic raw score', value: data?.authentic_score },
  ].filter(s => s.value != null);

  return (
    <Card title="Authenticity Assessment" icon={Shield}>
      {cfg ? <div className={`inline-flex items-center gap-3 px-6 py-3 rounded-xl border ${cfg.bg} mb-4`}>
        <Shield className={`w-6 h-6 ${cfg.color}`} />
        <span className={`text-xl font-bold ${cfg.color}`}>{cfg.text}</span>
      </div> : <p className="text-sm text-text-muted mb-4">Prediction unavailable</p>}
      {scores.length > 0 && (
        <div className="grid grid-cols-2 gap-3 mb-4">
          <p className="col-span-2 text-xs text-text-muted">The larger raw score selects the tag; these scores are not calibrated probabilities.</p>
          {scores.map(s => (
            <div key={s.name} className="bg-surface/50 rounded-xl p-4 text-center">
              <p className="text-2xl font-bold text-text">{Number(s.value).toFixed(3)}</p>
              <p className="text-xs text-text-muted mt-1">{s.name}</p>
            </div>
          ))}
        </div>
      )}
      {data?.model_name && <p className="text-xs text-text-muted">Model: {data.model_name}</p>}
    </Card>
  );
}

/* ────── EVIDENCE SECTION ────── */
function EvidenceSection({ data, explanation }: { data: any; explanation: any }) {
  if (!explanation && !data) return null;
  const hasModelPrediction = Boolean(data?.items?.some((item: any) => item.type === 'model_evidence'));
  const visibleSections = (explanation?.sections || []).filter((section: any) => section.title !== 'AI Model Assessment');
  const visibleItems = (data?.items || []).filter((item: any) => item.type !== 'model_evidence');
  return (
    <Card title="Supporting Evidence" icon={HelpCircle}>
      {hasModelPrediction
        ? <p className="text-sm text-text-muted mb-4">The tag above follows the comparison of the AI-generated and authentic raw scores. Other signals below provide context.</p>
        : explanation?.summary && <p className="text-sm text-text-muted mb-4">{explanation.summary}</p>}
      {explanation?.key_findings?.length > 0 && (
        <div className="space-y-2 mb-4">
          <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Key Findings</p>
          {explanation.key_findings.map((f: string, i: number) => (
            <div key={i} className="flex gap-2 text-sm"><Zap className="w-4 h-4 text-primary-light shrink-0 mt-0.5" /><span className="text-text-muted">{f}</span></div>
          ))}
        </div>
      )}
      {visibleSections.map((s: any, i: number) => (
        <div key={i} className="mb-3">
          <p className="text-sm font-medium text-text mb-1">{s.title}</p>
          <p className="text-xs text-text-muted">{s.content}</p>
        </div>
      ))}
      {data?.has_conflicting_evidence && (
        <div className="mt-3 flex items-center gap-2 text-yellow-400 text-xs">
          <AlertTriangle className="w-4 h-4" /> Conflicting evidence detected — assessment reliability reduced.
        </div>
      )}
      {visibleItems.length > 0 ? (
        <div className="mt-4 space-y-2">
          <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Evidence items</p>
          {visibleItems.map((item: any) => (
            <div key={item.evidence_id || `${item.type}-${item.source}-${item.description}`} className="rounded-lg bg-surface/60 p-3">
              <div className="flex flex-wrap items-center gap-2"><span className="text-sm font-medium text-text">{item.type || 'Evidence'}</span><StatusBadge status={item.status || 'completed'} /><span className="text-xs text-text-muted">{item.source}</span></div>
              <p className="text-xs text-text-muted mt-1">{item.description}</p>
            </div>
          ))}
        </div>
      ) : <p className="mt-3 text-sm text-text-muted">No evidence items were available.</p>}
    </Card>
  );
}

function PipelineDetails({ stages, warnings }: { stages?: Record<string, any>; warnings?: string[] }) {
  const entries = Object.entries(stages || {});
  return (
    <Card title="Pipeline status" icon={Info}>
      {warnings?.map((warning, i) => <p key={i} className="text-xs text-warning mb-2">{warning}</p>)}
      {entries.length ? <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
        {entries.map(([name, stage]) => <div key={name} className="rounded-lg bg-surface/60 p-3">
          <div className="flex items-center justify-between gap-2"><span className="text-sm text-text">{name.replace(/_/g, ' ')}</span><StatusBadge status={stage?.status || 'unavailable'} /></div>
          {stage?.runtime_ms != null && <p className="text-xs text-text-muted mt-1">{stage.runtime_ms} ms</p>}
          {stage?.error && <p className="text-xs text-warning mt-1">{stage.error}</p>}
          {name === 'preprocessing' && stage?.result?.original_width && <p className="text-xs text-text-muted mt-1">Input {stage.result.original_width} × {stage.result.original_height}px · EXIF orientation {stage.result.orientation_corrected ? 'corrected' : 'unchanged'} · resized variants: {(stage.result.resized_variants || []).join(', ') || 'none'}</p>}
        </div>)}
      </div> : <p className="text-sm text-text-muted">Pipeline stage details unavailable.</p>}
    </Card>
  );
}

function ForensicsSection({ forensics, quality, objects, statuses }: { forensics: any; quality: any; objects: any; statuses?: any }) {
  return <Card title="Forensics and visual analysis" icon={Eye}>
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <div><p className="text-xs uppercase text-text-muted mb-1">Image quality</p>{quality ? <pre className="text-xs text-text-muted whitespace-pre-wrap">{JSON.stringify(quality, null, 2)}</pre> : <p className="text-xs text-text-muted">{statuses?.quality || 'Unavailable'}</p>}</div>
      <div><p className="text-xs uppercase text-text-muted mb-1">Forensic signals</p>{forensics?.signals?.length ? <>{forensics.signals.map((s: any, i: number) => <p key={i} className="text-xs text-text-muted">{s.name} [{s.status}]{s.score != null ? ` · indicator ${Number(s.score).toFixed(3)}` : ''}: {s.interpretation}</p>)}<p className="text-[10px] text-text-muted mt-2">Indicators are encoding and content dependent; they are not manipulation probabilities.</p>{forensics.signals.find((s: any) => s.details?.artifact_url)?.details?.artifact_url && <a className="text-xs text-primary" href={forensics.signals.find((s: any) => s.details?.artifact_url).details.artifact_url} target="_blank" rel="noreferrer">View ELA visualization</a>}</> : <p className="text-xs text-text-muted">{statuses?.forensics || 'Unavailable'}</p>}</div>
      <div><p className="text-xs uppercase text-text-muted mb-1">Detected objects</p>{objects?.objects?.length ? objects.objects.map((o: any, i: number) => <p key={i} className="text-xs text-text-muted">{o.class_name || o.label || 'Object'}{o.confidence != null ? ` (${pct(o.confidence)})` : ''}</p>) : <p className="text-xs text-text-muted">{statuses?.vision || 'Unavailable or none detected'}</p>}</div>
    </div>
  </Card>;
}

/* ────── IMAGE VIEWER ────── */
function ImageViewer({ imageUrl, heatmapUrl, objects, xai }: { imageUrl?: string; heatmapUrl?: string; objects?: any; xai?: any }) {
  const [view, setView] = useState<'original' | 'heatmap'>('original');
  const [heatmapOk, setHeatmapOk] = useState(true);
  return (
    <Card title="Image Analysis" icon={Eye}>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setView('original')} className={clsx('px-4 py-2 rounded-lg text-sm font-medium transition-colors', view === 'original' ? 'bg-primary text-white' : 'bg-surface/50 text-text-muted hover:text-text')}>Original</button>
        {heatmapUrl && heatmapOk && <button onClick={() => setView('heatmap')} className={clsx('px-4 py-2 rounded-lg text-sm font-medium transition-colors', view === 'heatmap' ? 'bg-primary text-white' : 'bg-surface/50 text-text-muted hover:text-text')}>Heatmap</button>}
      </div>
      <div className="rounded-xl overflow-hidden bg-surface border border-border">
        {view === 'original' && imageUrl && (objects?.status === 'completed' ? <div className="w-full max-h-[500px] overflow-auto"><svg role="img" aria-label="Uploaded image with detected object boxes" viewBox={`0 0 ${objects.image_width || 1} ${objects.image_height || 1}`} className="block w-full max-h-[500px]" style={{ aspectRatio: `${objects.image_width || 1} / ${objects.image_height || 1}` }}><image href={imageUrl} width={objects.image_width || 1} height={objects.image_height || 1} preserveAspectRatio="none" />{(objects.objects || []).map((object: any, i: number) => { const [x1, y1, x2, y2] = object.bbox || []; if (![x1, y1, x2, y2].every(Number.isFinite)) return null; return <g key={i}><rect x={x1} y={y1} width={Math.max(0, x2-x1)} height={Math.max(0, y2-y1)} fill="none" stroke="#22d3ee" strokeWidth={Math.max(2, objects.image_width/300)} /><title>{object.class_name} ({pct(object.confidence)})</title></g>; })}</svg></div> : <img src={imageUrl} alt="Uploaded" className="w-full max-h-[500px] object-contain" />)}
        {view === 'heatmap' && heatmapUrl && <img src={heatmapUrl} alt="XAI Heatmap" className="w-full max-h-[500px] object-contain" onError={() => setHeatmapOk(false)} />}
        {!imageUrl && <div className="h-64 flex items-center justify-center text-text-muted">No image available</div>}
      </div>
      {view === 'heatmap' && <div className="mt-3 text-xs text-text-muted"><p>Grad-CAM attribution highlights image regions that influenced the selected model output. It does not show tampering or prove authenticity.</p>{xai?.target_class && <p className="mt-1">Target class: {xai.target_class}{xai.target_score != null ? ` · model score ${pct(xai.target_score)}` : ''}</p>}{xai?.suspicious_regions?.length > 0 && <p className="mt-1">High-activation regions (pixel boxes): {xai.suspicious_regions.slice(0, 5).map((region: any) => `[${(region.bbox || []).join(', ')}]`).join('; ')}</p>}</div>}
    </Card>
  );
}

function ReportSummary({ data }: { data: any }) {
  if (!data) return null;
  const summary = data.summary || data;
  const authenticityPrediction = data.authenticity?.model_prediction;
  const displayValue = (title: string, value: any) => {
    if (title === 'Authenticity') {
      const prediction = authenticityPrediction || value?.model_prediction;
      if (prediction === 'likely_ai_generated') return 'AI-Generated';
      if (prediction === 'likely_authentic') return 'Real / Authentic';
      const aiScore = data.authenticity?.ai_generated_score;
      const authenticScore = data.authenticity?.authentic_score;
      if (typeof aiScore === 'number' && typeof authenticScore === 'number') {
        return aiScore > authenticScore ? 'AI-Generated' : 'Real / Authentic';
      }
    }
    return value?.classification || value?.status || value?.assessment_mode || 'Unavailable';
  };
  const entries = [
    ['Authenticity', summary.authenticity],
    ['Factual claims', summary.factual_claims],
    ['Provenance', summary.provenance],
    ['Spread', summary.spread],
    ['Potential social impact', summary.potential_social_impact],
  ];
  return <Card title="Analysis Summary" icon={FileText}>
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
      {entries.map(([title, value]: any) => <div key={title} className="rounded-lg bg-surface/60 p-3">
        <p className="text-xs uppercase font-semibold text-text-muted">{title}</p>
        <p className="text-sm text-text mt-1">{displayValue(title, value)}{value?.verdicts?.length ? ` · ${value.verdicts.join(', ')}` : ''}{value?.categories?.length ? ` · ${value.categories.join(', ')}` : ''}</p>
        {value?.note && <p className="text-xs text-text-muted mt-1">{value.note}</p>}
      </div>)}
    </div>
  </Card>;
}

/* ────── OCR SECTION ────── */
function OcrSection({ data }: { data: any }) {
  if (!data) return (
    <Card title="OCR — Extracted Text" icon={FileText}>
      <p className="text-sm text-text-muted">OCR result unavailable: this report contains no OCR result.</p>
    </Card>
  );
  return (
    <Card title="OCR — Extracted Text" icon={FileText}>
      <div className="flex gap-4 mb-3 text-xs text-text-muted"><span>Status: <span className="text-text">{data.status || 'unknown'}</span></span><span>Engine: <span className="text-text">{data.engine || '—'}</span></span><span>Languages: <span className="text-text">{(data.language_configuration || []).join(', ') || '—'}</span></span></div>
      {data.status === 'failed' || data.status === 'unavailable' ? <p className="text-sm text-text-muted">{data.no_text_reason || `OCR unavailable. ${(data.warnings || []).join(' ')}`}</p> : data.has_text ? (
        <>
          <div className="flex gap-4 mb-3">
            <span className="text-xs text-text-muted">Script estimate: <span className="text-text font-medium">{data.language || '—'}</span></span>
            <span className="text-xs text-text-muted">Words: <span className="text-text font-medium">{data.word_count || 0}</span></span>
            <span className="text-xs text-text-muted">Confidence: <span className="text-text font-medium">{pct(data.total_confidence)}</span></span>
          </div>
          <div className="bg-surface rounded-xl p-4 font-mono text-sm text-text-muted whitespace-pre-wrap border border-border max-h-48 overflow-y-auto">
            {data.text || data.normalized_text || '(empty)'}
          </div>
        </>
      ) : <p className="text-sm text-text-muted">{data.no_text_reason || 'OCR ran but found no readable text regions. The image may contain no text or the text may be too small, blurry, or low-contrast to recognize.'}</p>}
    </Card>
  );
}

/* ────── METADATA SECTION ────── */
function MetadataSection({ data, imageInfo }: { data: any; imageInfo: any }) {
  const rows: [string, any][] = [];
  if (imageInfo) {
    if (imageInfo.width) rows.push(['Dimensions', `${imageInfo.width} × ${imageInfo.height} px`]);
    if (imageInfo.format) rows.push(['Format', imageInfo.format]);
    if (imageInfo.file_size_bytes) rows.push(['File Size', `${(imageInfo.file_size_bytes / 1024).toFixed(1)} KB`]);
    if (imageInfo.mime_type) rows.push(['MIME Type', imageInfo.mime_type]);
  }
  if (data) {
    if (data.has_exif != null) rows.push(['EXIF Data', data.has_exif ? 'Present' : 'Not found']);
    if (data.camera_model) rows.push(['Camera', data.camera_model]);
    if (data.software) rows.push(['Software', data.software]);
    if (data.gps_available) rows.push(['GPS', 'Available (not displayed for privacy)']);
  }
  if (rows.length === 0) return null;
  return (
    <Card title="Image Metadata" icon={Info}>
      <div className="space-y-2">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between text-sm py-1.5 border-b border-border/50 last:border-0">
            <span className="text-text-muted">{k}</span><span className="text-text font-medium">{String(v)}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ────── FACT CHECK SECTION ────── */
function safeSourceUrl(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'https:' ? parsed.href : null;
  } catch { return null; }
}

function ClaimsSection({ data }: { data: any }) {
  if (!data) return null;
  const claims = data.claims || [];
  return (
    <Card title="OCR Claims" icon={Search} badge={<StatusBadge status={data.status || 'completed'} />}>
      {claims.length === 0 && <p className="text-sm text-text-muted">No candidate claims were extracted from the image text.</p>}
      <div className="space-y-3">
        {claims.map((claim: any) => (
          <div key={claim.claim_id || claim.original_text} className="bg-surface rounded-xl p-4 border border-border">
            <p className="text-sm text-text font-medium">{claim.original_text || claim.claim}</p>
            {claim.normalized_text && claim.normalized_text !== (claim.original_text || claim.claim) && <p className="text-xs text-text-muted mt-1">Normalized: {claim.normalized_text}</p>}
            <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs text-text-muted">
              <span>Type: {claim.classification || claim.verification_eligibility || claim.claim_type || 'unknown'}</span>
              <span>Language: {claim.language || 'unknown'}{claim.language_detection_status === 'script_estimate' ? ' · script estimate' : ''}</span>
              {claim.ocr_confidence != null && <span>OCR confidence: {Math.round(claim.ocr_confidence * 100)}%</span>}
            </div>
            {claim.eligibility_reason && <p className="text-xs text-text-muted mt-1">{claim.eligibility_reason}</p>}
          </div>
        ))}
      </div>
    </Card>
  );
}

function FactCheckSection({ data }: { data: any }) {
  if (!data) return null;
  return (
    <Card title="Fact-Check Results" icon={Search} badge={<StatusBadge status={data.status} />}>
      {data.status === 'unavailable' && <p className="text-sm text-text-muted">Fact-check API not configured. Set <code className="bg-surface px-1.5 py-0.5 rounded text-xs">FACTCHECK_API_KEY</code> or <code className="bg-surface px-1.5 py-0.5 rounded text-xs">GOOGLE_API_KEY</code> in .env.</p>}
      {data.status === 'failed' && <p className="text-sm text-danger">{data.error || 'API request failed.'}</p>}
      {(data.status === 'no_results' || data.status === 'partial') && <p className="text-sm text-text-muted">No conclusive review was available for some or all eligible claims.</p>}
      {data.claim_verifications?.length > 0 && (
        <div className="space-y-3 mb-4">
          {data.claim_verifications.map((v: any) => (
            <div key={v.claim_id || v.original_claim} className="bg-surface rounded-xl p-4 border border-border">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-1"><span className="text-sm font-semibold text-text">{v.verdict?.replace(/_/g, ' ') || 'UNVERIFIABLE'}</span><span className="text-xs text-text-muted">{v.verification_status || ''}</span></div>
              <p className="text-sm text-text">{v.original_claim}</p>
              <p className="text-xs text-text-muted mt-1">{v.reasoning_summary}</p>
              {v.limitations?.length > 0 && <p className="text-xs text-text-muted mt-1">{v.limitations.join(' ')}</p>}
            </div>
          ))}
        </div>
      )}
      {(data.status === 'success' || data.status === 'partial') && data.claims?.length > 0 && (
        <div className="space-y-3">
          {data.claims.map((c: any, i: number) => (
            <div key={i} className="bg-surface rounded-xl p-4 border border-border">
              <p className="text-xs text-text-muted mb-1">Third-party review · {c.claim_match === 'strong_match' ? 'strong text match' : 'related search result; claim match unconfirmed'}</p>
              <p className="text-sm text-text font-medium mb-1">{c.title || c.claimed_text || 'Fact-check review'}</p>
              {c.claimed_text && <p className="text-xs text-text-muted">Reviewed statement: “{c.claimed_text}”</p>}
              <p className="text-xs text-text-muted">Rating: <span className="font-semibold text-warning">{c.textual_rating || '—'}</span></p>
              <p className="text-xs text-text-muted">Publisher: {c.publisher_name || '—'} {c.review_date && `· ${c.review_date}`} {safeSourceUrl(c.url) && <a href={safeSourceUrl(c.url)!} target="_blank" rel="noopener noreferrer" className="text-primary-light hover:underline ml-2">Source →</a>}</p>
            </div>
          ))}
        </div>
      )}
      {data.note && <p className="mt-3 text-xs text-text-muted italic">{data.note}</p>}
    </Card>
  );
}

/* ────── PROVENANCE SECTION ────── */
function ProvenanceSection({ data }: { data: any }) {
  if (!data) return null;
  return (
    <Card title="Web Provenance" icon={Globe} badge={<StatusBadge status={data.status} />}>
      {data.status === 'unavailable' && <p className="text-sm text-text-muted">Web detection API not configured. Set <code className="bg-surface px-1.5 py-0.5 rounded text-xs">GOOGLE_API_KEY</code> in .env.</p>}
      {data.status === 'failed' && <p className="text-sm text-danger">{data.error || 'API request failed.'}</p>}
      {data.status === 'no_results' && <p className="text-sm text-text-muted">No web matches found for this image.</p>}
      {data.status === 'success' && (
        <>
          {data.best_guess_labels?.length > 0 && <p className="text-sm text-text mb-3">Web suggests: <span className="font-medium">{data.best_guess_labels.join(', ')}</span></p>}
          {data.matching_pages?.length > 0 && (
            <div className="mb-3">
              <p className="text-xs font-semibold text-text-muted uppercase mb-2">Matching Pages ({data.matching_pages.length})</p>
              {data.matching_pages.slice(0, 5).map((p: any, i: number) => (
                <div key={i} className="text-xs text-text-muted py-1.5 border-b border-border/50 last:border-0">
                  {safeSourceUrl(p.url) ? <a href={safeSourceUrl(p.url)!} target="_blank" rel="noopener noreferrer" className="text-primary-light hover:underline truncate block">{p.page_title || p.url}</a> : <span className="truncate block">{p.page_title || 'External result URL unavailable'}</span>}
                  <span className="text-xs text-text-muted">Match: {p.match_type}</span>
                </div>
              ))}
            </div>
          )}
          {data.similar_images?.length > 0 && <div className="space-y-1 text-xs text-text-muted"><p className="font-semibold uppercase">Similar images</p>{data.similar_images.slice(0, 5).map((item: any, i: number) => <div key={i}>{safeSourceUrl(item.url) ? <a href={safeSourceUrl(item.url)!} target="_blank" rel="noopener noreferrer" className="text-primary-light hover:underline">Open result</a> : 'Result URL unavailable'} <span>· {item.match_type}</span></div>)}</div>}
          {data.content_credentials && <p className="text-xs text-text-muted mt-3">Content Credentials: {data.content_credentials.status === 'unavailable' ? 'Not assessed' : data.content_credentials.status}. {data.content_credentials.reason || ''}</p>}
        </>
      )}
      {data.note && <p className="mt-3 text-xs text-yellow-400/70 italic">{data.note}</p>}
    </Card>
  );
}

function HashSection({ hashes, imageInfo }: { hashes: any; imageInfo: any }) {
  const values = hashes || {};
  const sha = values.sha256 || imageInfo?.sha256_hash;
  const perceptual = values.phash || imageInfo?.perceptual_hash;
  if (!sha && !perceptual) return null;
  return <Card title="Image Hashes" icon={Info}>
    <div className="space-y-2 text-xs">
      {sha && <div><p className="text-text-muted">SHA-256 · exact file-byte fingerprint</p><code className="text-text break-all">{sha}</code></div>}
      {perceptual && <div><p className="text-text-muted">pHash · visual similarity hint</p><code className="text-text break-all">{perceptual}</code></div>}
      <p className="text-text-muted">A perceptual-hash match can indicate visually similar files; it does not establish identical content or authenticity.</p>
    </div>
  </Card>;
}

/* ────── VIRALITY SECTION ────── */
function ViralitySection({ data }: { data: any }) {
  if (!data) return null;
  const riskCfg: Record<string, { color: string; label: string }> = {
    low: { color: 'text-green-400', label: 'Low Risk' },
    moderate: { color: 'text-yellow-400', label: 'Moderate Risk' },
    high: { color: 'text-red-400', label: 'High Risk' },
    unknown: { color: 'text-text-muted', label: 'Unknown' },
  };
  const risk = riskCfg[data.risk_level] || riskCfg.unknown;
  const features = (data.features || []).filter((f: any) => f.available && f.present && f.value != null);
  const chartData = features.map((f: any) => ({ name: f.name.replace(/_/g, ' '), value: Math.round(f.value * 100) }));

  return (
    <Card title="Content Spread Indicators (Heuristic)" icon={TrendingUp} badge={<StatusBadge status={data.data_status} />}>
      {data.data_status === 'DEMO' && (
        <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3 mb-4">
          <p className="text-sm text-purple-400 font-medium">⚠️ DEMO DATA — NOT A REAL-WORLD PREDICTION</p>
        </div>
      )}
      <div className="flex items-center gap-4 mb-4">
        <div className="text-center">
          <p className={`text-3xl font-bold ${risk.color}`}>{data.virality_score == null ? 'Unavailable' : pct(data.virality_score)}</p>
          <p className={`text-sm font-medium ${risk.color}`}>{data.virality_score == null ? 'No assessed features' : 'Indicator level: ' + risk.label}</p>
        </div>
        {chartData.length > 0 && (
          <div className="flex-1 h-32">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical">
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                <YAxis type="category" dataKey="name" width={90} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', fontSize: '12px' }} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {chartData.map((_: any, i: number) => <Cell key={i} fill={i % 2 ? '#818cf8' : '#6366f1'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
      {data.disclaimer && <p className="text-xs text-yellow-400/70 italic">{data.disclaimer}</p>}
    </Card>
  );
}

/* ────── REGIONAL MAP ────── */
const INDIA_CENTER: [number, number] = [22.5937, 78.9629];
const STATE_COORDS: Record<string, [number, number]> = {
  "Andhra Pradesh": [15.9129, 79.7400],
  "Arunachal Pradesh": [28.2180, 94.7278],
  "Assam": [26.2006, 92.9376],
  "Bihar": [25.0961, 85.3131],
  "Chhattisgarh": [21.2787, 81.8661],
  "Goa": [15.2993, 74.1240],
  "Gujarat": [22.2587, 71.1924],
  "Haryana": [29.0588, 76.0856],
  "Himachal Pradesh": [31.1048, 77.1734],
  "Jharkhand": [23.6102, 85.2799],
  "Karnataka": [15.3173, 75.7139],
  "Kerala": [10.8505, 76.2711],
  "Madhya Pradesh": [22.9734, 78.6569],
  "Maharashtra": [19.7515, 75.7139],
  "Manipur": [24.6637, 93.9063],
  "Meghalaya": [25.4670, 91.3662],
  "Mizoram": [23.1645, 92.9376],
  "Nagaland": [26.1584, 94.5624],
  "Odisha": [20.9517, 85.0985],
  "Punjab": [31.1471, 75.3412],
  "Rajasthan": [27.0238, 74.2179],
  "Sikkim": [27.5330, 88.5122],
  "Tamil Nadu": [11.1271, 78.6569],
  "Telangana": [18.1124, 79.0193],
  "Tripura": [23.9408, 91.9882],
  "Uttar Pradesh": [26.8467, 80.9462],
  "Uttarakhand": [30.0668, 79.0193],
  "West Bengal": [22.9868, 87.8550],
  "Delhi": [28.7041, 77.1025],
  "Jammu and Kashmir": [33.7782, 76.5762]
};

function RegionalSection({ data }: { data: any }) {
  if (!data) return null;
  const topRegions = (data.regions || []).filter((r: any) => r.risk_score != null).slice(0, 10);
  const chartData = topRegions.map((r: any) => ({ name: r.region, score: Math.round(r.risk_score * 100) }));

  const getColor = (score: number) => {
    if (score > 60) return '#ef4444';
    if (score > 30) return '#f59e0b';
    return '#22c55e';
  };

  return (
    <Card title="India — Regional Dissemination Data" icon={MapPin} badge={<StatusBadge status={data.data_status} />}>
      {data.data_status === 'DEMO' && (
        <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3 mb-4">
          <p className="text-sm text-purple-400 font-medium">⚠️ DEMO DATA — NOT A REAL-WORLD PREDICTION</p>
        </div>
      )}
      {data.data_status === 'DATA_UNAVAILABLE' && (
        <div className="bg-surface/50 border border-border rounded-lg p-3 mb-4">
          <p className="text-sm text-text-muted">Regional scores and rankings are unavailable because traceable state and union territory dissemination data is not configured.</p>
        </div>
      )}
      {data.content_language && <p className="text-sm text-text-muted mb-3">Content language: <span className="font-medium text-text">{data.content_language}</span></p>}
      {data.top_risk_regions?.length > 0 && (
        <p className="text-sm text-text-muted mb-4">Top regions: <span className="font-medium text-text">{data.top_risk_regions.join(', ')}</span></p>
      )}
      
      {chartData.length > 0 && <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[400px]">
        {chartData.length > 0 && (
          <div className="h-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ left: 10 }}>
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 11, fill: '#94a3b8' }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', fontSize: '12px' }} formatter={(v: any) => [`${v}%`, 'Risk Score']} />
                <Bar dataKey="score" radius={[0, 6, 6, 0]}>
                  {chartData.map((d: any, i: number) => <Cell key={i} fill={getColor(d.score)} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        
        <div className="h-full rounded-xl overflow-hidden border border-border bg-surface relative z-0">
          <MapContainer center={INDIA_CENTER} zoom={4} className="h-full w-full bg-[#0f172a]" zoomControl={false}>
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              attribution='&copy; CartoDB'
            />
            {data.regions?.map((r: any) => {
              const coords = STATE_COORDS[r.region];
              if (!coords) return null;
              if (r.risk_score == null || r.confidence == null) return null;
              const radius = Math.max(5, r.risk_score * 20);
              const color = getColor(r.risk_score * 100);
              return (
                <CircleMarker 
                  key={r.region} 
                  center={coords} 
                  radius={radius}
                  pathOptions={{ color, fillColor: color, fillOpacity: 0.6, weight: 1 }}
                >
                  <MapTooltip className="bg-surface-2 border-border text-text">
                    <div className="text-sm">
                      <p className="font-bold mb-1">{r.region}</p>
                      <p>Risk Score: {(r.risk_score * 100).toFixed(1)}%</p>
                      <p>Confidence: {(r.confidence * 100).toFixed(1)}%</p>
                      {r.evidence?.map((e: string, i: number) => <p key={i} className="text-xs text-text-muted mt-1">{e}</p>)}
                    </div>
                  </MapTooltip>
                </CircleMarker>
              );
            })}
          </MapContainer>
        </div>
      </div>}
      
      {data.disclaimer && <p className="mt-4 text-xs text-yellow-400/70 italic">{data.disclaimer}</p>}
    </Card>
  );
}

/* ────── SOCIAL IMPACT SECTION ────── */
function SocialImpactSection({ data }: { data: any }) {
  if (!data) return null;
  const sevCfg: Record<string, string> = { low: 'text-green-400', moderate: 'text-yellow-400', high: 'text-red-400', unknown: 'text-text-muted', unassessed: 'text-text-muted', indicators_present: 'text-yellow-400', no_indicators_detected: 'text-text-muted' };
  const categories = data.categories || [];
  const radarData: any[] = [];

  return (
    <Card title="Potential Harm Content Categories" icon={Users} badge={<StatusBadge status={data.data_status} />}>
      {data.data_status === 'DEMO' && (
        <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3 mb-4">
          <p className="text-sm text-purple-400 font-medium">⚠️ DEMO DATA — NOT A REAL-WORLD PREDICTION</p>
        </div>
      )}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <div className="bg-surface/50 rounded-xl p-3 text-center">
          <p className={`text-lg font-bold ${sevCfg[data.risk_level] || 'text-text-muted'}`}>{(data.risk_level || 'unknown').toUpperCase()}</p>
          <p className="text-xs text-text-muted">Observed harm</p>
        </div>
        <div className="bg-surface/50 rounded-xl p-3 text-center">
          <p className={`text-lg font-bold ${sevCfg[data.misinformation_severity] || 'text-text-muted'}`}>{(data.misinformation_severity || 'unknown').toUpperCase()}</p>
          <p className="text-xs text-text-muted">Falsity severity</p>
        </div>
        <div className="bg-surface/50 rounded-xl p-3 text-center">
          <p className="text-lg font-bold text-text">{(data.estimated_audience_scale || 'unknown').toUpperCase()}</p>
          <p className="text-xs text-text-muted">Audience reach</p>
        </div>
        <div className="bg-surface/50 rounded-xl p-3 text-center">
          <p className={`text-lg font-bold ${sevCfg[data.urgency_level] || 'text-text-muted'}`}>{(data.urgency_level || 'low').toUpperCase()}</p>
          <p className="text-xs text-text-muted">Urgency</p>
        </div>
      </div>
      {radarData.length > 0 && (
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData}>
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 9, fill: '#64748b' }} />
              <Radar dataKey="score" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}
      {data.primary_concerns?.length > 0 && (
        <div className="mt-3 space-y-1">
          <p className="text-xs font-semibold text-text-muted uppercase">Primary Concerns</p>
          {data.primary_concerns.map((c: string, i: number) => (
            <div key={i} className="flex gap-2 text-xs text-warning"><AlertOctagon className="w-3.5 h-3.5 shrink-0 mt-0.5" />{c}</div>
          ))}
        </div>
      )}
      {categories.filter((c: any) => c.indicators_found > 0).map((c: any) => <div key={c.category} className="mt-2 text-xs text-text-muted"><strong>{c.category.replace(/_/g, ' ')}:</strong> indicators {c.indicators.join(', ') || 'none'}; claims {c.supporting_claim_ids?.join(', ') || 'not linked'}; verification {c.verification_statuses?.join(', ') || 'unassessed'}.</div>)}
      {data.disclaimer && <p className="mt-3 text-xs text-yellow-400/70 italic">{data.disclaimer}</p>}
    </Card>
  );
}

/* ────── LIMITATIONS SECTION ────── */
function LimitationsSection({ limitations, disclaimer }: { limitations?: string[]; disclaimer?: string }) {
  return (
    <Card title="Limitations & Disclaimer" icon={AlertTriangle} className="border-yellow-500/20">
      {disclaimer && <p className="text-sm text-yellow-400/80 mb-4 leading-relaxed">{disclaimer}</p>}
      {limitations?.length ? (
        <ul className="space-y-2">
          {limitations.map((l, i) => <li key={i} className="text-xs text-text-muted flex gap-2"><span className="text-yellow-400 shrink-0">•</span>{l}</li>)}
        </ul>
      ) : null}
    </Card>
  );
}

/* ═══════════════ MAIN APP ═══════════════ */
export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [uploading, setUploading] = useState(false);
  const [analysisId, setAnalysisId] = useState<string>('');
  const [status, setStatus] = useState<any>(null);
  const [report, setReport] = useState<any>(null);
  const [error, setError] = useState<string>('');
  const [dragOver, setDragOver] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  /* ── File select ── */
  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith('image/')) { setError('Please select an image file.'); return; }
    if (f.size > 20 * 1024 * 1024) { setError('File too large (max 20 MB).'); return; }
    setFile(f); setError(''); setReport(null); setStatus(null); setAnalysisId('');
    sessionStorage.removeItem('truthlens.analysisId');
    window.history.replaceState({}, '', window.location.pathname);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  }, []);

  /* ── Upload ── */
  const handleUpload = useCallback(async () => {
    if (!file) return;
    setUploading(true); setError('');
    try {
      const result = await uploadImage(file);
      setAnalysisId(result.analysis_id);
      sessionStorage.setItem('truthlens.analysisId', result.analysis_id);
      window.history.replaceState({}, '', `?analysis=${encodeURIComponent(result.analysis_id)}`);
        setStatus(result);
      startPolling(result.analysis_id);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || 'Upload failed.');
    } finally { setUploading(false); }
  }, [file]);

  /* ── Poll ── */
  const startPolling = (id: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await getAnalysisStatus(id);
        setStatus(s);
        if (['completed', 'partial', 'failed'].includes(s.status)) {
          clearInterval(pollRef.current!); pollRef.current = null;
          if (s.status === 'completed' || s.status === 'partial' || s.status === 'failed') {
            const r = await getAnalysisReport(id);
            if (r.analysis_id !== id) throw new Error('The report did not match the submitted analysis.');
            setReport(r);
          }
        }
      } catch (e: any) { console.error('Poll error:', e); setError(e.response?.data?.detail || e.message || 'Unable to retrieve the analysis report.'); }
    }, 2000);
  };

  // Restore the submitted analysis when a user refreshes the results page.
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get('analysis') || sessionStorage.getItem('truthlens.analysisId');
    if (!id || !/^[0-9a-f-]{36}$/i.test(id)) return;
    setAnalysisId(id);
    getAnalysisStatus(id).then(async s => {
      setStatus(s);
      if (['completed', 'partial', 'failed'].includes(s.status)) setReport(await getAnalysisReport(id));
      else startPolling(id);
    }).catch((e: any) => setError(e.response?.data?.detail || 'Saved analysis is unavailable or has expired.'));
    // The analysis ID is intentionally read once on page load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  /* ── Drag & Drop ── */
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const isProcessing = status?.status === 'processing' || status?.status === 'queued';
  const isCompleted = ['completed', 'partial', 'failed'].includes(status?.status) && report;
  const isFailed = status?.status === 'failed';

  return (
    <div className="min-h-screen">
      {/* ── HEADER ── */}
      <header className="border-b border-border/50 bg-surface/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-primary-dark flex items-center justify-center">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-primary-light to-primary bg-clip-text text-transparent">TruthLens</h1>
              <p className="text-xs text-text-muted">AI-Assisted Image Verification & Social Impact Analysis</p>
            </div>
          </div>
          {report?.processing_time_ms && <span className="text-xs text-text-muted">{(report.processing_time_ms / 1000).toFixed(1)}s</span>}
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        {/* ── UPLOAD ZONE ── */}
        {!isCompleted && (
          <div
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            className={clsx(
              'border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 cursor-pointer',
              dragOver ? 'border-primary bg-primary/5 scale-[1.01]' : 'border-border hover:border-primary/50',
              file ? 'bg-surface-2/50' : 'bg-surface-2/30',
            )}
            onClick={() => !file && inputRef.current?.click()}
          >
            <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
            {!file ? (
              <div className="space-y-3">
                <Upload className="w-12 h-12 text-primary-light mx-auto" />
                <p className="text-lg font-medium text-text">Drop an image here or click to select</p>
                <p className="text-sm text-text-muted">Supports JPEG, PNG, WebP, GIF, BMP, TIFF — Max 20 MB</p>
              </div>
            ) : (
              <div className="flex flex-col sm:flex-row items-center gap-6">
                {preview && <img src={preview} alt="Preview" className="w-40 h-40 object-cover rounded-xl border border-border" />}
                <div className="text-left space-y-1">
                  <p className="text-sm font-medium text-text">{file.name}</p>
                  <p className="text-xs text-text-muted">{(file.size / 1024).toFixed(1)} KB · {file.type}</p>
                </div>
                <div className="flex gap-3 ml-auto">
                  <button onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(''); setAnalysisId(''); setStatus(null); setReport(null); }} className="px-4 py-2 rounded-lg bg-surface border border-border text-sm text-text-muted hover:text-text transition-colors">Clear</button>
                  <button onClick={(e) => { e.stopPropagation(); handleUpload(); }} disabled={uploading || isProcessing} className="px-6 py-2 rounded-lg bg-gradient-to-r from-primary to-primary-dark text-white text-sm font-medium hover:opacity-90 disabled:opacity-50 transition-all flex items-center gap-2">
                    {(uploading || isProcessing) && <Loader2 className="w-4 h-4 animate-spin" />}
                    {uploading ? 'Uploading…' : isProcessing ? 'Analyzing…' : 'Analyze'}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {error && <div className="bg-danger/10 border border-danger/30 text-danger rounded-xl p-4 text-sm">{error}</div>}

        {/* ── PROGRESS ── */}
        {isProcessing && status && (
          <Card title="Analysis Progress" icon={Loader2}>
            <ProgressTracker completed={status.progress?.completed_steps || status.completed_steps || []} current={status.progress?.current_step || status.current_step} />
            <p className="mt-3 text-xs text-text-muted">Progress reflects completed backend stages. Optional model initialization or external services can extend runtime.</p>
          </Card>
        )}

        {/* ── FAILED ── */}
        {isFailed && !report && (
          <Card title="Analysis Failed" icon={XCircle}>
            <p className="text-sm text-danger">{status?.error_message || 'Pipeline error.'}</p>
            <button onClick={() => { setFile(null); setPreview(''); setAnalysisId(''); setStatus(null); setReport(null); }} className="mt-4 px-4 py-2 rounded-lg bg-surface border border-border text-sm text-text-muted hover:text-text">Try Again</button>
          </Card>
        )}

        {/* ═══ REPORT ═══ */}
        {isCompleted && report && (
          <>
            {report.status !== 'completed' && <div className="rounded-xl border border-warning/30 bg-warning/10 p-4 text-sm text-warning">Analysis status: {report.status}. Review unavailable or failed stages below; this assessment may be incomplete.</div>}
            <ReportSummary data={report} />
            {/* New analysis button */}
            <div className="flex justify-end">
              <button onClick={() => { setFile(null); setPreview(''); setAnalysisId(''); sessionStorage.removeItem('truthlens.analysisId'); window.history.replaceState({}, '', window.location.pathname); setStatus(null); setReport(null); }} className="px-4 py-2 rounded-lg bg-surface-2 border border-border text-sm text-text-muted hover:text-text transition-colors flex items-center gap-2">
                <Upload className="w-4 h-4" /> New Analysis
              </button>
            </div>

            {/* 1. Authenticity */}
            <AuthenticitySection data={report.authenticity} />

            {/* 2. Evidence / Why */}
            <EvidenceSection data={report.evidence} explanation={report.explanation} />
            <PipelineDetails stages={report.stages} warnings={report.warnings} />

            {/* 3. Image Viewer */}
            <ImageViewer imageUrl={analysisId ? getAnalysisImageUrl(analysisId) : preview} heatmapUrl={analysisId && report.xai?.status === 'completed' ? getHeatmapUrl(analysisId) : undefined} xai={report.xai} objects={{...(report.objects?.status === 'completed' ? report.objects : {}), image_width: report.objects?.image_width || report.image?.width || 1, image_height: report.objects?.image_height || report.image?.height || 1}} />

            {/* 4. OCR */}
            <OcrSection data={report.ocr} />

            {/* 5. Metadata */}
            <MetadataSection data={report.metadata} imageInfo={report.image} />
            <ForensicsSection forensics={report.forensics} quality={report.quality} objects={report.objects} statuses={report.module_statuses} />

            {/* 6. Fact Check */}
            <ClaimsSection data={report.claims} />
            <FactCheckSection data={report.fact_checks} />

            {/* 7. Provenance */}
            <ProvenanceSection data={report.provenance} />
            <HashSection hashes={report.hashes} imageInfo={report.image} />

            {/* Two-column layout for virality & social impact */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* 8. Virality */}
              <ViralitySection data={report.virality} />
              {/* 9. Social Impact */}
              <SocialImpactSection data={report.social_impact} />
            </div>

            {/* 10. Regional Spread */}
            <RegionalSection data={report.regional_spread} />

            {/* 11. Limitations */}
            <LimitationsSection limitations={report.limitations} disclaimer={report.disclaimer} />
          </>
        )}
      </main>

      <footer className="border-t border-border/50 mt-16 py-6 text-center">
        <p className="text-xs text-text-muted">TruthLens — AI-Assisted Image Verification System · This is a research prototype, not a definitive truth detector.</p>
      </footer>
    </div>
  );
}
