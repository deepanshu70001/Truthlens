import { useState, useCallback, useEffect, useRef } from 'react';
import {
  Shield, Upload, AlertTriangle, CheckCircle, XCircle, Clock, Loader2,
  Eye, FileText, Globe, Search, TrendingUp, MapPin, Users, Info,
  ChevronDown, ChevronUp, Zap, AlertOctagon, HelpCircle
} from 'lucide-react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { MapContainer, TileLayer, GeoJSON, Tooltip as MapTooltip, CircleMarker } from 'react-leaflet';
import { uploadImage, getAnalysisStatus, getAnalysisReport, getHeatmapUrl } from './services/api';
import './index.css';

/* ────── helpers ────── */
const pct = (v?: number) => v != null ? `${(v * 100).toFixed(1)}%` : '—';
const clsx = (...cls: (string | false | undefined)[]) => cls.filter(Boolean).join(' ');

const STEPS = [
  'Upload', 'Preprocessing', 'Authenticity', 'Forensics', 'OCR',
  'Context', 'XAI', 'Text Norm', 'Claims', 'Fact Check',
  'Provenance', 'Hashing', 'Evidence', 'Fusion', 'Explanation',
  'Virality', 'Regional', 'Social Impact', 'Report'
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
    success: { color: 'bg-green-500/20 text-green-400 border-green-500/30', label: 'Available' },
    failed: { color: 'bg-red-500/20 text-red-400 border-red-500/30', label: 'Failed' },
    no_results: { color: 'bg-gray-500/20 text-gray-400 border-gray-500/30', label: 'No Results' },
  };
  const c = cfg[status] || cfg['unavailable'];
  return <span className={`text-xs px-2.5 py-1 rounded-full border ${c.color}`}>{c.label}</span>;
}

/* ────── COLLAPSIBLE SECTION ────── */
function Section({ title, icon: Icon, children, defaultOpen = true, badge }: {
  title: string; icon?: any; children: React.ReactNode; defaultOpen?: boolean; badge?: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Card title={title} icon={Icon} badge={badge}>
      <button onClick={() => setOpen(!open)} className="absolute top-6 right-6 text-text-muted hover:text-text">
        {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      {open && children}
    </Card>
  );
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
function AuthenticitySection({ data, assessment }: { data: any; assessment: any }) {
  const label = assessment?.label || data?.label || 'inconclusive';
  const labelCfg: Record<string, { color: string; bg: string; text: string }> = {
    likely_authentic: { color: 'text-green-400', bg: 'bg-green-500/10 border-green-500/30', text: 'Likely Authentic' },
    likely_ai_generated: { color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/30', text: 'Likely AI-Generated' },
    likely_manipulated: { color: 'text-orange-400', bg: 'bg-orange-500/10 border-orange-500/30', text: 'Likely Manipulated' },
    inconclusive: { color: 'text-yellow-400', bg: 'bg-yellow-500/10 border-yellow-500/30', text: 'Inconclusive' },
  };
  const cfg = labelCfg[label] || labelCfg.inconclusive;
  const scores = [
    { name: 'AI-Generated', value: data?.ai_generated_score },
    { name: 'Manipulation', value: data?.manipulation_score },
    { name: 'Authentic', value: data?.authentic_score },
  ].filter(s => s.value != null);

  return (
    <Card title="Authenticity Assessment" icon={Shield}>
      <div className={`inline-flex items-center gap-3 px-6 py-3 rounded-xl border ${cfg.bg} mb-4`}>
        <Shield className={`w-6 h-6 ${cfg.color}`} />
        <span className={`text-xl font-bold ${cfg.color}`}>{cfg.text}</span>
      </div>
      {assessment?.confidence != null && (
        <p className="text-sm text-text-muted mb-4">Confidence: {pct(assessment.confidence)}
          <span className="ml-2 text-xs text-yellow-400">(uncalibrated)</span>
        </p>
      )}
      {scores.length > 0 && (
        <div className="grid grid-cols-3 gap-3 mb-4">
          {scores.map(s => (
            <div key={s.name} className="bg-surface/50 rounded-xl p-4 text-center">
              <p className="text-2xl font-bold text-text">{pct(s.value)}</p>
              <p className="text-xs text-text-muted mt-1">{s.name}</p>
            </div>
          ))}
        </div>
      )}
      {data?.model_name && <p className="text-xs text-text-muted">Model: {data.model_name}</p>}
      {assessment?.caveat && <p className="mt-3 text-xs text-yellow-400/80 italic">{assessment.caveat}</p>}
    </Card>
  );
}

/* ────── EVIDENCE SECTION ────── */
function EvidenceSection({ data, explanation }: { data: any; explanation: any }) {
  if (!explanation && !data) return null;
  return (
    <Card title="Why This Assessment?" icon={HelpCircle}>
      {explanation?.summary && <p className="text-sm text-text-muted mb-4">{explanation.summary}</p>}
      {explanation?.key_findings?.length > 0 && (
        <div className="space-y-2 mb-4">
          <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Key Findings</p>
          {explanation.key_findings.map((f: string, i: number) => (
            <div key={i} className="flex gap-2 text-sm"><Zap className="w-4 h-4 text-primary-light shrink-0 mt-0.5" /><span className="text-text-muted">{f}</span></div>
          ))}
        </div>
      )}
      {explanation?.sections?.map((s: any, i: number) => (
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
    </Card>
  );
}

/* ────── IMAGE VIEWER ────── */
function ImageViewer({ imageUrl, heatmapUrl }: { imageUrl?: string; heatmapUrl?: string }) {
  const [view, setView] = useState<'original' | 'heatmap'>('original');
  const [heatmapOk, setHeatmapOk] = useState(true);
  return (
    <Card title="Image Analysis" icon={Eye}>
      <div className="flex gap-2 mb-4">
        <button onClick={() => setView('original')} className={clsx('px-4 py-2 rounded-lg text-sm font-medium transition-colors', view === 'original' ? 'bg-primary text-white' : 'bg-surface/50 text-text-muted hover:text-text')}>Original</button>
        {heatmapUrl && heatmapOk && <button onClick={() => setView('heatmap')} className={clsx('px-4 py-2 rounded-lg text-sm font-medium transition-colors', view === 'heatmap' ? 'bg-primary text-white' : 'bg-surface/50 text-text-muted hover:text-text')}>Heatmap</button>}
      </div>
      <div className="rounded-xl overflow-hidden bg-surface border border-border">
        {view === 'original' && imageUrl && <img src={imageUrl} alt="Uploaded" className="w-full max-h-[500px] object-contain" />}
        {view === 'heatmap' && heatmapUrl && <img src={heatmapUrl} alt="XAI Heatmap" className="w-full max-h-[500px] object-contain" onError={() => setHeatmapOk(false)} />}
        {!imageUrl && <div className="h-64 flex items-center justify-center text-text-muted">No image available</div>}
      </div>
    </Card>
  );
}

/* ────── OCR SECTION ────── */
function OcrSection({ data }: { data: any }) {
  if (!data) return null;
  return (
    <Card title="OCR — Extracted Text" icon={FileText}>
      {data.has_text ? (
        <>
          <div className="flex gap-4 mb-3">
            <span className="text-xs text-text-muted">Language: <span className="text-text font-medium">{data.language || '—'}</span></span>
            <span className="text-xs text-text-muted">Words: <span className="text-text font-medium">{data.word_count || 0}</span></span>
            <span className="text-xs text-text-muted">Confidence: <span className="text-text font-medium">{pct(data.total_confidence)}</span></span>
          </div>
          <div className="bg-surface rounded-xl p-4 font-mono text-sm text-text-muted whitespace-pre-wrap border border-border max-h-48 overflow-y-auto">
            {data.text || data.normalized_text || '(empty)'}
          </div>
        </>
      ) : <p className="text-sm text-text-muted">No text detected in the image.</p>}
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
function FactCheckSection({ data }: { data: any }) {
  if (!data) return null;
  return (
    <Card title="Fact-Check Results" icon={Search} badge={<StatusBadge status={data.status} />}>
      {data.status === 'unavailable' && <p className="text-sm text-text-muted">Fact-check API not configured. Set <code className="bg-surface px-1.5 py-0.5 rounded text-xs">FACTCHECK_API_KEY</code> or <code className="bg-surface px-1.5 py-0.5 rounded text-xs">GOOGLE_API_KEY</code> in .env.</p>}
      {data.status === 'failed' && <p className="text-sm text-danger">{data.error || 'API request failed.'}</p>}
      {data.status === 'no_results' && <p className="text-sm text-text-muted">No matching fact-checks found for extracted claims.</p>}
      {data.status === 'success' && data.claims?.length > 0 && (
        <div className="space-y-3">
          {data.claims.map((c: any, i: number) => (
            <div key={i} className="bg-surface rounded-xl p-4 border border-border">
              <p className="text-sm text-text font-medium mb-1">"{c.claimed_text}"</p>
              <p className="text-xs text-text-muted">Rating: <span className="font-semibold text-warning">{c.textual_rating || '—'}</span></p>
              <p className="text-xs text-text-muted">Publisher: {c.publisher_name || '—'} {c.url && <a href={c.url} target="_blank" rel="noopener" className="text-primary-light hover:underline ml-2">Source →</a>}</p>
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
                  <a href={p.url} target="_blank" rel="noopener" className="text-primary-light hover:underline truncate block">{p.page_title || p.url}</a>
                  <span className="text-xs text-text-muted">Match: {p.match_type}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
      {data.note && <p className="mt-3 text-xs text-yellow-400/70 italic">{data.note}</p>}
    </Card>
  );
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
  const features = (data.features || []).filter((f: any) => f.present);
  const chartData = features.map((f: any) => ({ name: f.name.replace(/_/g, ' '), value: Math.round(f.value * 100) }));

  return (
    <Card title="Estimated Spread Risk" icon={TrendingUp} badge={<StatusBadge status={data.data_status} />}>
      {data.data_status === 'DEMO' && (
        <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3 mb-4">
          <p className="text-sm text-purple-400 font-medium">⚠️ DEMO DATA — NOT A REAL-WORLD PREDICTION</p>
        </div>
      )}
      <div className="flex items-center gap-4 mb-4">
        <div className="text-center">
          <p className={`text-3xl font-bold ${risk.color}`}>{pct(data.virality_score)}</p>
          <p className={`text-sm font-medium ${risk.color}`}>{risk.label}</p>
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
  const topRegions = (data.regions || []).slice(0, 10);
  const chartData = topRegions.map((r: any) => ({ name: r.region, score: Math.round(r.risk_score * 100) }));

  const getColor = (score: number) => {
    if (score > 60) return '#ef4444';
    if (score > 30) return '#f59e0b';
    return '#22c55e';
  };

  return (
    <Card title="India — Estimated Regional Spread Risk" icon={MapPin} badge={<StatusBadge status={data.data_status} />}>
      {data.data_status === 'DEMO' && (
        <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3 mb-4">
          <p className="text-sm text-purple-400 font-medium">⚠️ DEMO DATA — NOT A REAL-WORLD PREDICTION</p>
        </div>
      )}
      {data.data_status === 'DATA_UNAVAILABLE' && (
        <div className="bg-surface/50 border border-border rounded-lg p-3 mb-4">
          <p className="text-sm text-text-muted">Regional spread prediction unavailable because validated regional training data is not configured. Scores are heuristics based on language compatibility.</p>
        </div>
      )}
      {data.content_language && <p className="text-sm text-text-muted mb-3">Content language: <span className="font-medium text-text">{data.content_language}</span></p>}
      {data.top_risk_regions?.length > 0 && (
        <p className="text-sm text-text-muted mb-4">Top regions: <span className="font-medium text-text">{data.top_risk_regions.join(', ')}</span></p>
      )}
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[400px]">
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
      </div>
      
      {data.disclaimer && <p className="mt-4 text-xs text-yellow-400/70 italic">{data.disclaimer}</p>}
    </Card>
  );
}

/* ────── SOCIAL IMPACT SECTION ────── */
function SocialImpactSection({ data }: { data: any }) {
  if (!data) return null;
  const sevCfg: Record<string, string> = { low: 'text-green-400', moderate: 'text-yellow-400', high: 'text-red-400', unknown: 'text-text-muted' };
  const categories = data.categories || [];
  const radarData = categories.map((c: any) => ({ subject: c.category.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()), score: Math.round(c.risk_score * 100) }));

  return (
    <Card title="Social Impact Assessment" icon={Users} badge={<StatusBadge status={data.data_status} />}>
      {data.data_status === 'DEMO' && (
        <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3 mb-4">
          <p className="text-sm text-purple-400 font-medium">⚠️ DEMO DATA — NOT A REAL-WORLD PREDICTION</p>
        </div>
      )}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <div className="bg-surface/50 rounded-xl p-3 text-center">
          <p className={`text-lg font-bold ${sevCfg[data.risk_level] || 'text-text-muted'}`}>{(data.risk_level || 'unknown').toUpperCase()}</p>
          <p className="text-xs text-text-muted">Overall Risk</p>
        </div>
        <div className="bg-surface/50 rounded-xl p-3 text-center">
          <p className={`text-lg font-bold ${sevCfg[data.misinformation_severity] || 'text-text-muted'}`}>{(data.misinformation_severity || 'unknown').toUpperCase()}</p>
          <p className="text-xs text-text-muted">Misinfo Severity</p>
        </div>
        <div className="bg-surface/50 rounded-xl p-3 text-center">
          <p className="text-lg font-bold text-text">{(data.estimated_audience_scale || 'unknown').toUpperCase()}</p>
          <p className="text-xs text-text-muted">Audience Scale</p>
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
        if (s.status === 'completed' || s.status === 'failed') {
          clearInterval(pollRef.current!); pollRef.current = null;
          if (s.status === 'completed') {
            const r = await getAnalysisReport(id);
            setReport(r);
          }
        }
      } catch (e: any) { console.error('Poll error:', e); }
    }, 2000);
  };

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  /* ── Drag & Drop ── */
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const isProcessing = status?.status === 'processing' || status?.status === 'queued';
  const isCompleted = status?.status === 'completed' && report;
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
            <ProgressTracker completed={status.completed_steps || []} current={status.current_step} />
            <p className="mt-3 text-xs text-text-muted">Processing… this may take 1-2 minutes for full analysis.</p>
          </Card>
        )}

        {/* ── FAILED ── */}
        {isFailed && (
          <Card title="Analysis Failed" icon={XCircle}>
            <p className="text-sm text-danger">{status?.error_message || 'Pipeline error.'}</p>
            <button onClick={() => { setFile(null); setPreview(''); setAnalysisId(''); setStatus(null); setReport(null); }} className="mt-4 px-4 py-2 rounded-lg bg-surface border border-border text-sm text-text-muted hover:text-text">Try Again</button>
          </Card>
        )}

        {/* ═══ REPORT ═══ */}
        {isCompleted && report && (
          <>
            {/* New analysis button */}
            <div className="flex justify-end">
              <button onClick={() => { setFile(null); setPreview(''); setAnalysisId(''); setStatus(null); setReport(null); }} className="px-4 py-2 rounded-lg bg-surface-2 border border-border text-sm text-text-muted hover:text-text transition-colors flex items-center gap-2">
                <Upload className="w-4 h-4" /> New Analysis
              </button>
            </div>

            {/* 1. Authenticity */}
            <AuthenticitySection data={report.authenticity} assessment={report.assessment} />

            {/* 2. Evidence / Why */}
            <EvidenceSection data={report.evidence} explanation={report.explanation} />

            {/* 3. Image Viewer */}
            <ImageViewer imageUrl={preview} heatmapUrl={analysisId ? getHeatmapUrl(analysisId) : undefined} />

            {/* 4. OCR */}
            <OcrSection data={report.ocr} />

            {/* 5. Metadata */}
            <MetadataSection data={report.metadata} imageInfo={report.image} />

            {/* 6. Fact Check */}
            <FactCheckSection data={report.fact_checks} />

            {/* 7. Provenance */}
            <ProvenanceSection data={report.provenance} />

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
